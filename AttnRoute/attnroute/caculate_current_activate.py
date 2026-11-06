from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any, Optional, Tuple
from util import repr_tools, nethook
import gc

def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    minus = None
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, minus=minus, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken="last", minus=None, **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()

def get_sentence_last_token_io_memory_efficient(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    sentences: List[str],
    module_template: str = "model.layers.{}.mlp",
    batch_size: int = 8,  # 减小批大小
    max_length: int = 128,  # 限制序列长度
    use_cache: bool = False,  # 禁用缓存以节省内存
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    内存优化的版本：分批处理，限制序列长度
    
    适用于Mistral等大模型
    """
    
    device = next(model.parameters()).device
    model.eval()  # 确保在评估模式
    
    all_input_reprs = []
    all_output_reprs = []
    
    # 分批处理句子
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(sentences)+batch_size-1)//batch_size}, "
              f"大小: {len(batch_sentences)}")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # 1. 对批次句子进行编码，限制长度
            inputs = tok(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # 2. 计算每个句子的最后一个有效token位置
            attention_mask = inputs['attention_mask']
            seq_lengths = attention_mask.sum(dim=1) - 1
            
            # 对于Mistral模型，通常不会自动添加eos_token
            last_token_positions = seq_lengths.tolist()
            
            # 3. 使用更节省内存的方式：直接获取hidden_states
            with torch.no_grad():
                # 禁用缓存以节省内存
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=use_cache,
                    return_dict=True
                )
            
            # 4. 根据模型类型确定如何获取模块输入输出
            model_type = model.config.model_type.lower()
            
            if 'mistral' in model_type:
                # Mistral模型结构：model.layers.{layer}.self_attn.o_proj
                # 我们需要分别获取self_attn.o_proj的输入和输出
                
                # 方法1：使用自定义钩子获取模块输入输出
                batch_input_reprs, batch_output_reprs = get_mistral_module_io(
                    model, layer, inputs, last_token_positions, module_template
                )
                
            else:
                # 通用方法：使用hidden_states
                # 注意：这只能获取层级的输入输出，不能获取具体模块
                
                # 第layer层的输入（即第layer-1层的输出，如果是第0层则是嵌入层输出）
                if layer == 0:
                    layer_inputs = outputs.hidden_states[0]
                else:
                    layer_inputs = outputs.hidden_states[layer]
                
                # 第layer层的输出
                layer_outputs = outputs.hidden_states[layer + 1]
                
                # 提取每个句子的最后一个token
                batch_input_reprs = []
                batch_output_reprs = []
                
                for j, pos in enumerate(last_token_positions):
                    batch_input_reprs.append(layer_inputs[j, pos])
                    batch_output_reprs.append(layer_outputs[j, pos])
                
                batch_input_reprs = torch.stack(batch_input_reprs)
                batch_output_reprs = torch.stack(batch_output_reprs)
            
            all_input_reprs.append(batch_input_reprs.cpu())  # 移到CPU
            all_output_reprs.append(batch_output_reprs.cpu())  # 移到CPU
            
            # 释放中间变量
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except torch.cuda.OutOfMemoryError:
            print(f"内存不足！尝试减小批次大小到 {batch_size//2}")
            # 递归减小批次大小
            if batch_size > 1:
                return get_sentence_last_token_io_memory_efficient(
                    model, tok, layer, sentences, 
                    module_template, batch_size//2, max_length
                )
            else:
                raise
    
    # 合并所有批次
    final_input_reprs = torch.cat(all_input_reprs, dim=0)
    final_output_reprs = torch.cat(all_output_reprs, dim=0)
    
    print(f"完成！获取 {len(sentences)} 个句子的表示，形状: {final_input_reprs.shape}")
    
    return final_input_reprs, final_output_reprs

def get_mistral_module_io(
    model: AutoModelForCausalLM,
    layer: int,
    inputs: dict,
    positions: List[int],
    module_template: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    专门为Mistral模型设计的模块输入输出获取函数
    使用前向钩子，避免nethook的内存开销
    """
    
    device = next(model.parameters()).device
    
    # 构建完整模块名
    module_name = module_template.format(layer)
    
    # 分解模块名以获取模块对象
    module_parts = module_name.split('.')
    current_module = model
    
    for part in module_parts:
        current_module = getattr(current_module, part)
    
    # 存储输入输出的钩子
    module_input = []
    module_output = []
    
    def input_hook(module, input):
        module_input.append(input[0].detach())  # 通常是元组，取第一个
    
    def output_hook(module, input, output):
        module_output.append(output.detach())
    
    # 注册钩子
    input_handle = current_module.register_forward_pre_hook(input_hook)
    output_handle = current_module.register_forward_hook(output_hook)
    
    try:
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
        
        # 检查是否成功获取
        if not module_input or not module_output:
            raise ValueError(f"未能获取模块 {module_name} 的输入输出")
        
        # 提取输入输出张量
        input_tensor = module_input[0]  # [batch, seq_len, hidden]
        output_tensor = module_output[0]  # [batch, seq_len, hidden]
        
        # 提取每个句子的指定位置
        batch_input_reprs = []
        batch_output_reprs = []
        
        for i, pos in enumerate(positions):
            batch_input_reprs.append(input_tensor[i, pos])
            batch_output_reprs.append(output_tensor[i, pos])
        
        batch_input_reprs = torch.stack(batch_input_reprs)
        batch_output_reprs = torch.stack(batch_output_reprs)
        
        return batch_input_reprs, batch_output_reprs
        
    finally:
        # 移除钩子
        input_handle.remove()
        output_handle.remove()


def get_sentence_last_token_io_lightweight(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    sentences: List[str],
    max_length: int = 128,
    batch_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    最轻量级的版本：仅获取层级的输入输出，不监控具体模块
    这是最节省内存的方法
    """
    
    device = next(model.parameters()).device
    model.eval()
    
    all_input_reprs = []
    all_output_reprs = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        
        # 编码
        inputs = tok(
            batch_sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # 获取序列长度
        attention_mask = inputs['attention_mask']
        seq_lengths = attention_mask.sum(dim=1) - 1
        
        # 前向传播，只获取hidden_states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False  # 禁用缓存节省内存
            )
        
        # 获取指定层的输入输出
        # 注意：这是层级的，不是模块级的
        if layer == 0:
            layer_inputs = outputs.hidden_states[0]  # 嵌入层输出
        else:
            layer_inputs = outputs.hidden_states[layer]  # 第layer-1层输出
        
        layer_outputs = outputs.hidden_states[layer + 1]  # 第layer层输出
        
        # 提取最后一个token
        batch_inputs = []
        batch_outputs = []
        
        for j, length in enumerate(seq_lengths):
            batch_inputs.append(layer_inputs[j, length])
            batch_outputs.append(layer_outputs[j, length])
        
        all_input_reprs.append(torch.stack(batch_inputs).cpu())
        all_output_reprs.append(torch.stack(batch_outputs).cpu())
        
        # 清理
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return torch.cat(all_input_reprs, dim=0), torch.cat(all_output_reprs, dim=0)

def get_sequence_full_io(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    sentences: List[str],
    module_template: str = "model.layers.{}.self_attn.o_proj",
    device: Optional[str] = None,
    batch_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    获取句子列表中每个句子的【完整序列】在指定模块的输入和输出表示。
    Designed for AttnRoute-style methods that score and mask each token.
    
    参数:
    ----------
    model : AutoModelForCausalLM
        目标语言模型
    tok : AutoTokenizer
        对应的分词器
    layer : int
        目标层编号
    sentences : List[str]
        句子列表
    module_template : str
        模块名称模板
    device : Optional[str]
        指定设备
    
    返回:
    ----------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        1. input_reprs:  [batch_size, seq_len, hidden_dim]
        2. output_reprs: [batch_size, seq_len, hidden_dim]
        3. attention_mask: [batch_size, seq_len] (用于后续区分 padding)
    """
    
    if device is None:
        device = next(model.parameters()).device
    if tok.padding_side != 'right':
        tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    all_inputs = []
    all_outputs = []
    all_masks = []
    total = len(sentences)
    print(f"正在获取层 {layer} 的全序列数据 (总数: {total})...")
    for i in range(0, total, batch_size):
        batch_sentences = sentences[i : i + batch_size]
        current_batch_size = len(batch_sentences)
        
        # 1. 显存清理 (关键!)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        try:
            # 2. 编码当前批次
            inputs = tok(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(device)

            if i == 0:
                print(f"DEBUG Check Tokens: {inputs['input_ids'][0, :5].tolist()}")
            
            # 3. 监控模块
            module_name = module_template.format(layer)
            
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=True,
                retain_output=True,
                detach=True # 必须分离梯度
            ) as tr:
                # 4. 前向传播 (必须用 no_grad 节省一半显存)
                with torch.no_grad():
                    model(**inputs)
            
            # 5. 提取数据
            # 处理 Input
            raw_in = tr.input
            input_tensor = raw_in[0] if isinstance(raw_in, tuple) else raw_in
            
            # 处理 Output
            raw_out = tr.output
            output_tensor = raw_out[0] if isinstance(raw_out, tuple) else raw_out
            
            # 6. 转移到 CPU 并保存 (防止显存堆积)
            # AttnRoute needs full-sequence activations, so store the tensor directly.
            #all_inputs.append(input_tensor.cpu())
            #all_outputs.append(output_tensor.cpu())
            #all_masks.append(inputs['attention_mask'].cpu())
            all_inputs.append(input_tensor.detach().to(device="cpu", dtype=torch.float16))
            all_outputs.append(output_tensor.detach().to(device="cpu", dtype=torch.float16))
            all_masks.append(inputs["attention_mask"].detach().to(device="cpu", dtype=torch.bool))
            
            print(f"  批次 {i//batch_size + 1}: {input_tensor.shape} -> CPU")

        except RuntimeError as e:
            if "out of memory" in str(e) or "CUBLAS" in str(e):
                print(f"  [警告] 批次 {i} 显存不足/CUBLAS错误，尝试减半批大小...")
                if batch_size <= 1:
                    raise e # 没法再减了，抛出异常
                
                # 递归调用自己，使用更小的 batch_size
                # 注意：这里只处理剩余的部分
                remaining_sentences = sentences[i:]
                sub_inputs, sub_outputs, sub_masks = get_sequence_full_io(
                    model, tok, layer, remaining_sentences, 
                    module_template, device, batch_size // 2
                )
                
                # 合并之前的和递归的结果
                if len(all_inputs) > 0:
                    return (
                        torch.cat(all_inputs + [sub_inputs], dim=0),
                        torch.cat(all_outputs + [sub_outputs], dim=0),
                        torch.cat(all_masks + [sub_masks], dim=0)
                    )
                else:
                    return sub_inputs, sub_outputs, sub_masks
            else:
                raise e

    # --- 合并结果 ---
    # 注意：不同批次的 Seq Len 可能不同 (因为 padding)，需要对齐
    # 找到最大长度
    max_len = max([t.shape[1] for t in all_inputs])
    
    final_inputs = []
    final_outputs = []
    final_masks = []
    
    # 填充对齐 (Pad to max length)
    for inp, out, msk in zip(all_inputs, all_outputs, all_masks):
        curr_len = inp.shape[1]
        if curr_len < max_len:
            pad_len = max_len - curr_len
            # Pad Input/Output (通常补0即可，因为有Mask)
            # F.pad format: (left, right, top, bottom, ...)
            # 对最后一维(dim)不变，对倒数第二维(seq)补 pad_len
            inp = torch.nn.functional.pad(inp, (0, 0, 0, pad_len))
            out = torch.nn.functional.pad(out, (0, 0, 0, pad_len))
            msk = torch.nn.functional.pad(msk, (0, pad_len))
            
        final_inputs.append(inp)
        final_outputs.append(out)
        final_masks.append(msk)
        
    final_inputs_t = torch.cat(final_inputs, dim=0)
    final_outputs_t = torch.cat(final_outputs, dim=0)
    final_masks_t = torch.cat(final_masks, dim=0)
    
    print(f"完成！最终形状: {final_inputs_t.shape}")
    return final_inputs_t, final_outputs_t, final_masks_t


