import torch
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from fastchat.model import get_conversation_template

import torch.nn.functional as F

from pathlib import Path
import util.nethook
from copy import deepcopy

from util import nethook
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns

from .utils import get_activations_at_layers
from rome.layer_stats import layer_stats

from .caculate_current_activate import get_module_input_output_at_words, get_sentence_last_token_io_memory_efficient, get_sequence_full_io

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
from util.globals import *
import json

import matplotlib.pyplot as plt
import numpy as np
import os
import random
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
    # 也修复其他可能的兼容性问题
    if not hasattr(np, 'Float'):
        np.Float = np.float64
    if not hasattr(np, 'Int'):
        np.Int = np.int64

def visualize_attention_diff_enhanced(model_orig, model_post, tok, prompt, layer_idx, title="Attention Decoupling", save_name=None, head_len=30, tail_len=30):
    """
    计算并可视化编辑前后的注意力差值矩阵 (Post - Pre)。
    默认模式：长程语义解耦 (提取最后的 tail_len 个 Token 对最前面的 head_len 个 Token 的注意力)
    """
    # ==========================================
    # 1. 防弹级净化：强制清洗与拼接 Prompt
    # ==========================================
    # 应对 Python 隐式将带逗号长字符串转为元组/列表的 Bug
    if isinstance(prompt, (list, tuple)):
        prompt = " ".join([str(p) for p in prompt])
    
    prompt = str(prompt)
    # 移除可能导致 Rust Tokenizer 崩溃的特殊 Unicode 代理对 (如 Emoji)
    prompt = prompt.encode('utf-8', 'ignore').decode('utf-8')
    prompt = prompt.replace(r"\ud83d\udd12", "[Normal]").replace(r"\ud83d\udd13", "[DevMode]")
    
    print(f"  [Diff Map] Processing prompt length: {len(prompt)} chars")
    
    # ==========================================
    # 2. 安全分词
    # ==========================================
    inputs = tok(prompt, return_tensors="pt").to(model_orig.device)
    token_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = [tok.decode([t]) for t in token_ids]
    
    # 防止文本太短不够截取
    actual_head = min(head_len, len(tokens))
    actual_tail = min(tail_len, len(tokens))
    
    # X轴标签：越狱设定区 (前 actual_head 个词)
    x_tokens = tokens[:actual_head]
    # Y轴标签：恶意指令区 (最后 actual_tail 个词)
    y_tokens = tokens[-actual_tail:]
    
    print(f"  [Diff Map] Extracting attention from models at layer {layer_idx}...")
    
    # ==========================================
    # 3. 提取注意力矩阵
    # ==========================================
    with torch.no_grad():
        # 提取原始模型注意力 [Seq_len, Seq_len]
        out_orig = model_orig(**inputs, output_attentions=True)
        attn_orig = out_orig.attentions[layer_idx][0].mean(dim=0).cpu().float().numpy()
        
        # 提取编辑后模型注意力 [Seq_len, Seq_len]
        out_post = model_post(**inputs, output_attentions=True)
        attn_post = out_post.attentions[layer_idx][0].mean(dim=0).cpu().float().numpy()

    # 计算整体差值矩阵
    diff_matrix = attn_post - attn_orig

    # ==========================================
    # 4. 提取非对称长程子矩阵
    # 行(Query): 取最后 actual_tail 个 (代表即将生成的决策点)
    # 列(Key): 取最前 actual_head 个 (代表前缀伪装角色)
    # ==========================================
    sub_matrix = diff_matrix[-actual_tail:, :actual_head]

    # ==========================================
    # 5. 高级绘图
    # ==========================================
    # 因为是不对称矩阵 (比如 30x15)，所以画布比例调整为扁长形
    plt.figure(figsize=(6, 6))
    
    # 计算色标的对称最大值
    v_limit = np.abs(sub_matrix).max()
    if v_limit < 1e-5: v_limit = 0.01 
    
    # 绘制热图
    ax=sns.heatmap(sub_matrix, xticklabels=x_tokens, yticklabels=y_tokens, 
                cmap="RdBu_r", center=0, vmin=-v_limit, vmax=v_limit,
                annot=False, fmt=".3f")
    plt.xticks(rotation=90,fontsize=12)
    plt.yticks(rotation=0,fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12) # 色带数字刻度放大
    cbar.set_label('Attention Change', size=12, weight='bold') # 色带标题放大
    
    if save_name:
        os.makedirs('optimization_plots', exist_ok=True)
        full_path = os.path.join('optimization_plots', f"{save_name}.pdf")
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"  [Diff Map] Long-range decoupling plot saved to: {full_path}")
        plt.close()
    else:
        plt.show()

def build_attnroute_subspace(
    model: torch.nn.Module,
    model_name,
    tok: Any,
    hparams: "AttnRouteHyperParams",
    device: str,
    all_target_layers: List[int],
    target_module_fmt: str = None,
    data_type: str = "jailbreak",
    eta: float = 0.5,           # 【新增】
    alpha_max: float = 2.5,
    rank_mode: str = "sve",
    fixed_rank: int = 1
) -> Dict[int, Dict[str, Any]]:
    print(">>>>Building NN-Enhanced Safety Subspace...<<<<")
    def format_prompt(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    
    # --- 1. 数据准备 ---
    print("Loading data...")
    if data_type == "jailbreak":
        print("Loading jailbreak data...")
        df = pd.read_csv("data/jailbreak_prompts.csv")
        prompts = df["prompt"].tolist()
    elif data_type == "harmful":
        print("Loading harmful data...")
        df = pd.read_csv("data/harmbench_behaviors_text_all.csv")
        prompts = df["Behavior"].tolist()
        targets = df["target"].tolist()
    
    path = "data/harmless.csv"
    df_harmless = pd.read_csv(path)[:200]
    harmless_prompts = df_harmless["prompt"].tolist()
    
    refusal_sentences = [
        "I cannot fulfill this request.",
        "I am unable to answer that question.",
        "I cannot assist with this specific task.",
        "I am sorry, but I can't help with that.",
        "I must decline this request.",
        "I am not able to comply with these instructions."
    ]

    compliance_sentences = [
        "I can certainly fulfill this request for you.",
        "I am happy to provide an answer to that question.",
        "I can definitely assist you with this specific task.",
        "Sure, I would be glad to help you with that.",
        "I will comply with these instructions immediately."
    ]
    augmented_refusals = [format_prompt(p) + random.choice(refusal_sentences) for p in prompts]
    augmented_compliance = [format_prompt(prompts[i]) + targets[i] for i in range(max(18, len(prompts)))]

    candidate_layers = sorted(list(set(all_target_layers)))
    print(f"Extracting activations for {len(candidate_layers)} layers...")
    
    harmless_acts = get_activations_at_layers(
        model, model_name, tok, harmless_prompts, 
        candidate_layers, device, module_template=target_module_fmt
    )
    harmful_acts = get_activations_at_layers(
        model, model_name, tok, prompts,
        candidate_layers, device, module_template=target_module_fmt
    )
    ref_acts_raw = get_activations_at_layers(model, model_name, tok, augmented_refusals, candidate_layers, device, add_special_tokens=False)
    comp_acts_raw = get_activations_at_layers(model, model_name, tok, augmented_compliance, candidate_layers, device, add_special_tokens=False)
    
    all_layers_assets = {}
    k_neighbors = 1  # 使用3个最近邻
    print(f"eta = {eta}")
    
    for layer_idx in candidate_layers:
        print(f"Processing Layer {layer_idx}...")
        
        Z_harm = harmful_acts[layer_idx].float().to(device)  # [N_harm, D]
        Z_safe = harmless_acts[layer_idx].float().to(device)  # [N_safe, D]
        
        # --- K近邻差异计算 ---
        print(f"  Computing {k_neighbors}-nearest neighbor differences...")
        nn_diffs = []
        
        for i in range(Z_harm.shape[0]):
            harm_vec = Z_harm[i:i+1]  # [1, D]
            
            # 计算到所有安全样本的距离
            distances = torch.cdist(harm_vec, Z_safe).squeeze()  # [N_safe]
            
            # 找到k个最近的安全样本
            _, nearest_indices = torch.topk(distances, 
                                          k=min(k_neighbors, Z_safe.shape[0]), 
                                          largest=False)
            nearest_safes = Z_safe[nearest_indices]  # [k, D]
            
            # 计算与k个最近邻的平均差异
            avg_nearest_safe = nearest_safes.mean(dim=0, keepdim=True)  # [1, D]
            diff = harm_vec - avg_nearest_safe  # [1, D]
            diff_augmented = diff
            
            nn_diffs.append(diff_augmented)
        
        # 堆叠所有差异
        Z_diff = torch.cat(nn_diffs, dim=0)  # [N_harm, D]
        
        # 组合差异
        Z_diff_enhanced = Z_diff
        
        # --- 标准化 ---
        Z_mean = Z_diff_enhanced.mean(dim=0)
        Z_std = Z_diff_enhanced.std(dim=0) + 1e-6
        Z_tilde = (Z_diff_enhanced - Z_mean) / Z_std
        
        # --- SVD 分解 ---
        U, S, V = torch.svd(Z_tilde)
        
        # --- 基于能量选择秩 ---
        squared_singular_values = S**2
        total_energy = squared_singular_values.sum()

        if rank_mode == "sve":
            p = squared_singular_values / total_energy
            log_p = torch.log(p + 1e-10)
            total_entropy = -(p * log_p).sum()
        
            r = 1
            current_cumulative_entropy = 0
            for i in range(len(p)):
                current_cumulative_entropy -= (p[i] * log_p[i])
                if (current_cumulative_entropy / total_entropy) > eta:
                    r = i + 1
                    break
        else:
            r = fixed_rank
            # 防止设定的 fixed_rank 超过实际维度
            r = min(r, len(S)) 
        
        print(f"  Layer {layer_idx}: SVE selected rank r = {r}")
        
        # --- 构建加权投影矩阵 ---
        sigma_1 = S[0]
        sigma_r = S[r-1]
        
        alphas = 1 + (alpha_max - 1) * (S[:r] - sigma_r) / (sigma_1 - sigma_r + 1e-6)
        alphas = torch.clamp(alphas, min=1.0)
        
        V_r = V[:, :r]
        V_r_weighted = V_r * torch.sqrt(alphas).unsqueeze(0)
        P_toxic = torch.matmul(V_r_weighted, V_r_weighted.T)
        
        Z_ref = ref_acts_raw[layer_idx].float().to(device)
        Z_comp = comp_acts_raw[layer_idx].float().to(device)
        
        # 对比减法：意图 A (拒绝) - 意图 B (顺从)
        # 这样抵消了 User Prompt 和 Assistant 格式的干扰
        #real_refusal_vec = (Z_ref - Z_comp).mean(dim=0)
        real_refusal_vec = Z_ref.mean(dim=0)
       
        harm_proj_original = torch.norm(Z_harm @ P_toxic, dim=1).mean()
        
        # 安全样本投影
        safe_proj = torch.norm(Z_safe @ P_toxic, dim=1).mean()
        
        separation_score = harm_proj_original / (safe_proj + 1e-8)
        
        all_layers_assets[layer_idx] = {
            'P_toxic': P_toxic,
            'refusal_vec': real_refusal_vec,
            'r': r,
            'snr_score': separation_score.item(),
            'method': 'nn_enhanced',
            'k_neighbors': k_neighbors
        }
        print(f"  Layer {layer_idx}: Rank={r}, Separation={separation_score.item():.4f}")
    
    return all_layers_assets

def cagrad_solve(grads, c=0.5):
    G = torch.stack([g.view(-1) for g in grads])
    device = G.device
    num_losses = G.shape[0]
    
    GG = torch.matmul(G, G.t()).cpu().numpy() # [num_losses, num_losses]
    g0_norm_sq = (torch.mean(G, dim=0).norm() ** 2).item() # ||g0||^2
    
    F = np.mean(GG, axis=1) # [num_losses]
    
    def obj_fn(w):
        # 目标: w.T @ GG @ w - (2/c) * w.T @ F
        term1 = np.dot(w, np.dot(GG, w))
        term2 = (2.0 / c) * np.dot(w, F)
        return term1 - term2

    def obj_jac(w):
        # 导数: 2 * GG @ w - (2/c) * F
        return 2 * np.dot(GG, w) - (2.0 / c) * F

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_losses))
    
    w0 = np.ones(num_losses) / num_losses
    
    res = minimize(obj_fn, w0, method='SLSQP', jac=obj_jac, 
                   bounds=bounds, constraints=constraints, tol=1e-6)
    
    w_opt = torch.from_numpy(res.x).float().to(device)
    
    final_grad = torch.matmul(w_opt, G)
    
    return final_grad


class AttnRouteProgram:
    def __init__(
        self,
        layer_assets: Dict[str, Any],
        hparams: "AttnRouteHyperParams",
        device: str = "cuda",
    ):
        print("Initializing AttnRouteProgram for a layer...")
        
        self.P_toxic = layer_assets['P_toxic'].to(device)
        self.refusal_vec = layer_assets['refusal_vec'].to(device)
        self.rank_r = layer_assets['r']
        self.alpha_erase = hparams.alpha
        self.beta_inject = hparams.beta
        self.maxdist = hparams.maxdist
        self.lr = hparams.lr
        self.device = device    

    def apply(self, x_in_original: torch.Tensor, x_out_original: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        orig_shape = x_out_original.shape
        x_k_cpu = x_in_original.reshape(-1, x_in_original.shape[-1]).to(device="cpu", dtype=torch.float16)
        x_v_cpu = x_out_original.reshape(-1, x_out_original.shape[-1]).to(device="cpu", dtype=torch.float16)
        if attn_mask is not None:
            mask_flat_cpu = attn_mask.reshape(-1, 1).to(device="cpu", dtype=torch.bool)
        else:
            mask_flat_cpu = torch.ones(x_k_cpu.shape[0], 1, dtype=torch.bool)
        final_v_cpu = torch.empty(
                x_v_cpu.shape,
                dtype=torch.float16,
                device="cpu",
        )
        device = self.P_toxic.device
        chunk_size = 512
        total_tokens = x_k_cpu.shape[0]
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk_k = x_k_cpu[start_idx:end_idx].to(device=device, dtype=torch.float32, non_blocking=True)
            chunk_v = x_v_cpu[start_idx:end_idx].to(device=device, dtype=torch.float32, non_blocking=True)
            chunk_mask = mask_flat_cpu[start_idx:end_idx].to(device=device, dtype=torch.float32, non_blocking=True)
            if chunk_mask.sum() == 0:
                final_v_cpu[start_idx:end_idx] = chunk_v.to(device="cpu", dtype=torch.float16)
                del chunk_k, chunk_v, chunk_mask
                torch.cuda.empty_cache()
                continue
            chunk_out = self._process_chunk(chunk_k, chunk_v, chunk_mask, device)
            final_v_cpu[start_idx:end_idx] = chunk_out.to(device="cpu", dtype=torch.float16)
            del chunk_k, chunk_v, chunk_mask, chunk_out
            torch.cuda.empty_cache()
        return final_v_cpu.reshape(orig_shape).to(dtype=x_out_original.dtype)

    def _process_chunk(self, x_k, x_v, mask, device):
        # --- Identify ---
        harm_proj = torch.matmul(x_v, self.P_toxic)
        score = torch.norm(harm_proj, p=2, dim=-1, keepdim=True)
        valid_scores = score[mask.bool()]
        score = score * mask
        if valid_scores.std() < 1e-6: # 防止除以0
            threshold = valid_scores.mean()
        else:
            threshold = valid_scores.mean() + 0.5 * valid_scores.std()
                
        erict_mask = (score > threshold).float() * mask
            
            # 如果这块里没有越狱指令，直接返回
        if erict_mask.sum() == 0:
            return x_v
            
            # --- Optimization ---
        target_v = x_v.clone().detach().requires_grad_(True)
        orig_v_detached = x_v.clone().detach()
        ref_vec_expanded = self.refusal_vec.unsqueeze(0).expand_as(target_v)

        # 1. 计算拒绝向量的模长
        ref_norm = torch.norm(ref_vec_expanded, p=2).item()
        
        print(f"DEBUG: Refusal Vector Norm = {ref_norm:.4f}")

        with torch.no_grad():
            init_harm_proj = torch.matmul(target_v, self.P_toxic)
            init_harm_norm = torch.norm(init_harm_proj, p=2, dim=-1)
            initial_clean = (erict_mask * init_harm_norm).sum()/(erict_mask.sum() + 1e-8)
            print(f"初始clean值: {initial_clean:.4f}")
        
        optimizer = optim.Adam([target_v], lr=self.lr)
        min_delta = 1e-2
        c_param = 0.5
        scale_clean = self.alpha_erase
        scale_align = self.beta_inject
        hard_limit = 100
        soft_limit = 50
        lookback_window = 5   # 回看过去 N 步来计算斜率
        clean_loss_history = []
        max_allowed_dist = self.maxdist
        history = {
            'step': [],
            'loss_clean': [],
            'loss_align': [],
            'grad_scale_ratio': [], # 记录动态缩放比例
            'dist': []
        }
        import time
        timestamp = int(time.time())
        for i in range(hard_limit):
            optimizer.zero_grad()
            # --- Loss Calculation (保持不变) ---
            sim = F.cosine_similarity(target_v, ref_vec_expanded, dim=-1).unsqueeze(-1)
            loss_align = (erict_mask * F.relu(0.9 - sim)).sum()/(erict_mask.sum() + 1e-8) 
            
            curr_harm_proj = torch.matmul(target_v, self.P_toxic)
            harm_norm = torch.norm(curr_harm_proj, p=2, dim=-1)
            epsilon = 1e-4
            stable_norm = torch.sqrt(harm_norm**2 + epsilon)
            loss_clean = (erict_mask * stable_norm).sum()/(erict_mask.sum() + 1e-8)
            clean_loss_history.append(loss_clean.item())

            history['step'].append(i)
            history['loss_clean'].append(loss_clean.item())
            history['loss_align'].append(loss_align.item())
            
            # 1. 计算原始梯度（此时不缩放损失）
            g_clean_raw = torch.autograd.grad(loss_clean, target_v, retain_graph=True)[0].view(-1)
            g_align_raw = torch.autograd.grad(loss_align, target_v, retain_graph=True)[0].view(-1)
            
            # 2. 计算原始梯度范数
            norm_clean = g_clean_raw.norm()
            norm_align = g_align_raw.norm()
            
            # 3. 动态设置缩放系数，目标是将两个梯度范数对齐到 `norm_clean` 的大小
            scale_clean = 1.0  # 基准，保持不变
            scale_align = (norm_clean / (norm_align + 1e-8)).detach()  # 关键：动态缩放
            history['grad_scale_ratio'].append(scale_align.item())
            
            # 4. 使用缩放后的系数重新计算有效梯度（等效于直接缩放梯度）
            g_clean = g_clean_raw * scale_clean
            g_align = g_align_raw * scale_align
            grads = [g_clean, g_align]
            final_grad = cagrad_solve(grads, c=c_param)
            if target_v.grad is None:
                target_v.grad = torch.zeros_like(target_v)
            target_v.grad = final_grad.view_as(target_v)
            optimizer.step()

            with torch.no_grad():
                # 计算当前的位移向量 delta
                delta = target_v - orig_v_detached
                
                # 计算每个 token 的位移距离
                # shape: [batch, seq, 1]
                dist_val = torch.norm(delta, p=2, dim=-1, keepdim=True)
                
                # 找到那些跑出圈的 token (dist > max)
                # 只有被 mask 选中的才需要约束，没选中的本来就不动
                mask_bool = erict_mask.bool()
                
                # 计算缩放系数：如果 dist <= max，系数为 1 (不动)；如果 dist > max，系数 < 1 (缩回)
                # clamp(max=1.0) 确保不会把本来就很近的向量强行拉远
                scale_factor = torch.clamp(max_allowed_dist / (dist_val + 1e-8), max=1.0)
                
                # 我们只修改那些越界的 token
                target_v.data = orig_v_detached + delta * scale_factor  
                real_delta = target_v - orig_v_detached
                real_dist_val = torch.norm(real_delta, p=2, dim=-1, keepdim=True)
                avg_dist = (real_dist_val * erict_mask).sum() / (erict_mask.sum() + 1e-8)
                
                history['dist'].append(avg_dist.item())

            if i % 5 == 0 or i == hard_limit - 1:
                with torch.no_grad():
                    mask_sum = erict_mask.sum().item() + 1e-8
                    curr_sim = (sim * erict_mask).sum().item() / mask_sum
                    curr_real_dist = (dist_val * erict_mask).sum().item() / mask_sum
                    curr_clean = (erict_mask * harm_norm).sum().item() / mask_sum
                    print(f"Step {i} | curr_clean: {curr_clean:.4f} | Harm Sim: {curr_sim:.4f} | Dist(Hard): {curr_real_dist:.4f}")
                    
                    if i >= soft_limit:
                        if len(clean_loss_history) > lookback_window:
                            recent_losses = clean_loss_history[-lookback_window:]
                            slope = (recent_losses[-1] - recent_losses[0]) / lookback_window
                            is_improving = slope < -1e-4

                            if is_improving:
                                print(f">>> [Auto-Extend] Loss is still dropping (Slope: {slope:.2e}). Continuing...")
                                continue    
                            else:
                                print(f">>> [Stop] Convergence reached (Plateau). Slope: {slope:.2e}")
                                break
                                
        #plot_optimization_curves(history, save_name=f"chunk_{timestamp}.png")
        return target_v.detach()
        

def smart_downsample(
    cur_input: torch.Tensor, 
    x_original: torch.Tensor, 
    x_safe: torch.Tensor, 
    max_anchors: int = 5000
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 确保所有操作都在 CPU 上进行，防止显存 OOM
    cur_input = cur_input.cpu()
    x_original = x_original.cpu()
    x_safe = x_safe.cpu()

    print(f"  [Smart Sample] Starting... Input shape: {cur_input.shape}")

    # 1. 计算每个Token的变化量 (L2 Norm)
    # diff shape: [Total_Tokens]
    diff = torch.norm(x_safe - x_original, p=2, dim=-1)
    
    # 2. 识别被修改的 Token (越狱特征)
    # 阈值 1e-4 用于过滤掉计算浮点误差
    changed_indices = torch.nonzero(diff > 1e-4).squeeze()
    
    # 维度修正：如果只有一个点被修改，squeeze 会变成 0 维，需要还原
    if changed_indices.dim() == 0 and changed_indices.numel() == 1:
        changed_indices = changed_indices.unsqueeze(0)
    elif changed_indices.numel() == 0:
        # 极少见情况：没有任何修改
        changed_indices = torch.tensor([], dtype=torch.long)
    
    # 3. 识别未修改的 Token (锚点候选)
    unchanged_indices = torch.nonzero(diff <= 1e-4).squeeze()
    
    # 4. 对锚点进行随机下采样
    if unchanged_indices.numel() > max_anchors:
        # 随机打乱并取前 max_anchors 个
        perm = torch.randperm(unchanged_indices.numel())
        selected_anchors = unchanged_indices[perm[:max_anchors]]
    else:
        selected_anchors = unchanged_indices
        
    # 5. 合并索引
    if changed_indices.numel() > 0:
        final_indices = torch.cat([changed_indices, selected_anchors]).long()
    else:
        print("  Warning: no tokens were modified by AttnRoute. Using only anchors.")
        final_indices = selected_anchors.long()
    
    print(f"  [Smart Sample] Filtered: {cur_input.shape[0]} -> {final_indices.numel()} "
          f"(Modified: {changed_indices.numel()}, Anchors: {selected_anchors.numel()})")
    
    # 6. 提取数据
    k_filtered = cur_input[final_indices]
    
    # 计算残差 = Target (Safe) - Original
    v_new_filtered = x_safe[final_indices]
    v_old_filtered = x_original[final_indices]
    residual_filtered = v_new_filtered - v_old_filtered
    num_changed = changed_indices.numel()
    if num_changed < residual_filtered.shape[0]:
        residual_filtered[num_changed:] = 0.0
    
    return k_filtered, residual_filtered

def apply_attnroute_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List,
    hparams: "AttnRouteHyperParams",
    safety_programs: Optional[Dict[int, "AttnRouteProgram"]],
    rewrite_module_tmp: str,
    layer_module_tmp: str,
    mom2_update_weight: float,
    target_layers: List[int],
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    edit_mode: str = "attnroute",
    norm_ref: Optional[Dict[str, float]] = None,
    stats_out_path: Optional[str] = None, 
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor], Dict[str, float]]:
    if copy:
        model = deepcopy(model)

    deltas, update_norms, ablation_stats = execute_attnroute_update(
        model, tok, requests, hparams, safety_programs,
        rewrite_module_tmp, layer_module_tmp, mom2_update_weight,
        target_layers, cache_template=cache_template,
        edit_mode=edit_mode, norm_ref=norm_ref,
    )
    if stats_out_path is not None:
        os.makedirs(os.path.dirname(stats_out_path), exist_ok=True)
        with open(stats_out_path, "w") as f:
            json.dump(ablation_stats, f, indent=2)

    weights_copy = {}
    with torch.no_grad():
        for w_name, (key_mat, _) in deltas.items():
            w = nethook.get_parameter(model, w_name)

            key_mat = key_mat.to(w.device).float()
            upd_matrix = upd_matrix_match_shape(key_mat, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone().cpu()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())} using AttnRoute.")
    return model, weights_copy, update_norms

def execute_attnroute_update(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: "AttnRouteHyperParams",
    safety_programs: Optional[Dict[int, "AttnRouteProgram"]],
    rewrite_module_tmp: str,
    layer_module_tmp: str,
    mom2_update_weight: float,
    target_layers: List[int],
    cache_template: Optional[str] = None,
    edit_mode: str = "attnroute",
    norm_ref: Optional[Dict[str, float]] = None,
):
    print(">>>> Executing AttnRoute precise weight update... <<<<")

    # ===== 消融统计（报告里要写） =====
    ablation_stats = {
        "cov_mode": getattr(hparams, "cov_mode", "full"),
        "n_layers": 0,
        "n_lstsq_fallback": 0,      # solve 失败退化为 lstsq 的层数
        "n_threshold_clipped": 0,   # 被 SAFE_THRESHOLD 截断的层数
        "n_singular_lhs": 0,        # lhs 条件数 > 1e10 的层数
    }

    deltas = {}
    update_norms = {}
    requests = deepcopy(requests)
    device = next(model.parameters()).device  # 获取 GPU 设备

    print(f"Target layers: {target_layers}")
    print(f"Covariance mode: {ablation_stats['cov_mode']}")

    for i, layer in enumerate(target_layers):
        weight_name = f"{rewrite_module_tmp.format(layer)}.weight"
        ablation_stats["n_layers"] += 1

        # ============================================================
        # random_direct 臂：完全不动，原样保留
        # ============================================================
        if edit_mode == "random_direct":
            print(f"\n--- Processing Layer {layer} [random_direct] ---")
            w = nethook.get_parameter(model, weight_name)
            if norm_ref is None or weight_name not in norm_ref:
                raise ValueError(f"Missing norm reference for {weight_name}")
            target_norm = float(norm_ref[weight_name])
            adj_k = torch.randn_like(w).float()
            adj_k = adj_k / (adj_k.norm() + 1e-8) * target_norm
            with torch.no_grad():
                delta_norm = adj_k.abs().mean()
                print(f"   Random direct raw norm: {adj_k.norm().item():.6f}")
                print(f"   Random direct delta mean(abs): {delta_norm.item():.6f}")

                if "o_proj" in rewrite_module_tmp:
                    SAFE_THRESHOLD = 0.05
                elif "v_proj" in rewrite_module_tmp:
                    SAFE_THRESHOLD = 0.05
                elif "down_proj" in rewrite_module_tmp:
                    SAFE_THRESHOLD = 0.05
                else:
                    SAFE_THRESHOLD = 0.01
                if delta_norm > SAFE_THRESHOLD:
                    scale = SAFE_THRESHOLD / delta_norm
                    print(f"   Explosion detected! Scaling down by {scale.item():.4f}")
                    adj_k = adj_k * scale
                    ablation_stats["n_threshold_clipped"] += 1
            update_norms[weight_name] = float(adj_k.norm().item())
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                torch.zeros(adj_k.shape[0], 1)
            )
            print(f"   Random direct update norm: {update_norms[weight_name]:.6f}")
            continue

        # ============================================================
        # attnroute 臂
        # ============================================================
        if safety_programs is None:
            raise ValueError("safety_programs must not be None when edit_mode != 'random_direct'")
        if layer not in safety_programs:
            raise KeyError(f"Missing AttnRouteProgram for layer {layer}")

        print(f"\n--- Processing Layer {layer} [cov_mode={ablation_stats['cov_mode']}] ---")

        # =======================================================
        # Stage 1: Get Original Activations (Full Sequence)
        # =======================================================
        print("1. Fetching full sequence activations (CPU)...")
        cur_input_cpu, cur_output_cpu, attn_mask_cpu = get_sequence_full_io(
            model, tok, layer, sentences=requests, module_template=layer_module_tmp
        )

        cur_input_flat = cur_input_cpu.reshape(-1, cur_input_cpu.shape[-1])
        cur_output_flat = cur_output_cpu.reshape(-1, cur_output_cpu.shape[-1])
        attn_mask_flat = attn_mask_cpu.reshape(-1)

        print(f"   Data Shape: {cur_input_flat.shape}")

        # =======================================================
        # Stage 2: Target Reconstruction
        # =======================================================
        print("2. Applying AttnRouteProgram (internal micro-batching)...")
        safety_program_for_layer = safety_programs[layer]
        x_safe_flat = safety_program_for_layer.apply(cur_input_flat, cur_output_flat, attn_mask_flat)

        if x_safe_flat.shape != cur_output_flat.shape:
            raise RuntimeError(f"Shape Mismatch! Safe: {x_safe_flat.shape}, Orig: {cur_output_flat.shape}")

        # =======================================================
        # Stage 3: Smart Downsampling & Residual
        # =======================================================
        print("3. Downsampling & Computing Residuals...")
        filtered_k, filtered_residual = smart_downsample(
            cur_input_flat,
            cur_output_flat,
            x_safe_flat,
            max_anchors=8192
        )

        del cur_input_cpu, cur_output_cpu, cur_input_flat, cur_output_flat, x_safe_flat, attn_mask_cpu
        import gc; gc.collect()

        # =======================================================
        # Stage 4: Solve for the Update Matrix
        # =======================================================
        print(f"4. Solving least-squares (Samples: {filtered_k.shape[0]})...")

        layer_ks = filtered_k.to(device).double()
        layer_targets = filtered_residual.to(device).double()

        layer_ks_t = layer_ks.T
        layer_targets_t = layer_targets.T

        # ---------- 4a. 按 cov_mode 决定是否加载 cov ----------
        cov = None
        if ablation_stats["cov_mode"] in ("full", "damped"):
            cov = get_cov(
                model, tok, rewrite_module_tmp.format(layer),
                hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype,
            ).to(device).double()
        else:
            # none 臂：明确跳过 get_cov，省时间 + 防 OOM
            print("   [cov_mode=none] Skipping get_cov().")

        # ---------- 4b. 构造 lhs ----------
        KtK = layer_ks_t @ layer_ks_t.T  # [D, D]

        if ablation_stats["cov_mode"] == "full":
            lhs = mom2_update_weight * cov + KtK
        elif ablation_stats["cov_mode"] == "none":
            lhs = KtK
        elif ablation_stats["cov_mode"] == "damped":
            eps = hparams.cov_damping * KtK.diag().mean()
            lhs = KtK + eps * torch.eye(KtK.shape[0], device=KtK.device, dtype=KtK.dtype)
        else:
            raise ValueError(f"Unknown cov_mode: {ablation_stats['cov_mode']}")

        # ---------- 4c. 记录条件数（诊断静默失效） ----------
        try:
            cond = torch.linalg.cond(lhs).item()
        except Exception:
            cond = float("inf")
        if cond > 1e10:
            ablation_stats["n_singular_lhs"] += 1
            print(f"   ⚠️ lhs ill-conditioned: cond={cond:.2e}")

        rhs = layer_ks_t @ layer_targets_t.T

        # ---------- 4d. solve + fallback ----------
        try:
            adj_k_transposed = torch.linalg.solve(lhs, rhs)
        except RuntimeError as e:
            print(f"   [cov_mode={ablation_stats['cov_mode']}] solve failed: {e}")
            print(f"   → Falling back to lstsq (minimum-norm solution).")
            adj_k_transposed = torch.linalg.lstsq(lhs, rhs).solution
            ablation_stats["n_lstsq_fallback"] += 1

        # 检查解是否含 NaN/Inf（近奇异时 lstsq 也可能给出 NaN）
        if not torch.isfinite(adj_k_transposed).all():
            print(f"   ⚠️ Non-finite solution detected; replacing with zeros for this layer.")
            adj_k_transposed = torch.zeros_like(adj_k_transposed)
            ablation_stats["n_lstsq_fallback"] += 1

        adj_k = adj_k_transposed.T.float()

        # ---------- 4e. SAFE_THRESHOLD 截断（none 臂也必须保留） ----------
        with torch.no_grad():
            delta_norm = adj_k.abs().mean()
            print(f"   Delta Norm: {delta_norm.item():.6f}")

            if "o_proj" in rewrite_module_tmp:
                SAFE_THRESHOLD = 0.05
            elif "v_proj" in rewrite_module_tmp:
                SAFE_THRESHOLD = 0.05
            elif "down_proj" in rewrite_module_tmp:
                SAFE_THRESHOLD = 0.05
            else:
                SAFE_THRESHOLD = 0.01

            if delta_norm > SAFE_THRESHOLD:
                scale = SAFE_THRESHOLD / delta_norm
                print(f"   ⚠️ Explosion detected! Scaling down by {scale.item():.4f}")
                adj_k = adj_k * scale
                ablation_stats["n_threshold_clipped"] += 1

        update_norms[weight_name] = float(adj_k.norm().item())
        deltas[weight_name] = (adj_k.detach().cpu(), torch.zeros(adj_k.shape[0], 1))

        print(f"   Final update norm: {update_norms[weight_name]:.6f}")
        print(f"   Layer {layer} Done.")

        # 清理 GPU
        del cov, layer_ks, layer_targets, layer_ks_t, layer_targets_t, lhs, rhs, adj_k, KtK
        torch.cuda.empty_cache()

    # ===== 打印消融统计（写进报告） =====
    print("\n===== Covariance Ablation Summary =====")
    for k, v in ablation_stats.items():
        print(f"  {k}: {v}")
    print("=======================================\n")

    print(f"AttnRoute updates computed for: {list(deltas.keys())}")
    return deltas, update_norms,ablation_stats

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AttnRoute does not match original weight shape. "
            "Check for bugs in the code?"
        )





