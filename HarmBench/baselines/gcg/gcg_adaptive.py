import gc
import json
import os
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from ..baseline import SingleBehaviorRedTeamingMethod
from ..model_utils import get_template
from ..check_refusal_utils import check_refusal_completions
from .gcg_utils import sample_control, get_nonascii_toks

# ============================== GCG CLASS DEFINITION ============================== #
class AdaptiveGCG(SingleBehaviorRedTeamingMethod):
    def __init__(self,
                 target_model,
                 targets_path,
                 num_steps=50,
                 adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                 allow_non_ascii=False,
                 search_width=512,
                 use_prefix_cache=True,
                 eval_steps=10,
                 eval_with_check_refusal=False,
                 check_refusal_min_loss=0.1,
                 early_stopping=False,
                 early_stopping_min_loss=0.1,
                 starting_search_batch_size=None,
                 # === 新增：自适应攻击防御感知参数 ===
                 p_toxic_dict=None,  # 字典格式 {layer_idx: P_toxic_tensor}
                 adaptive_alpha=0.1, # 规避防御机制的惩罚权重
                 **model_kwargs):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        ...
        """
        super().__init__(target_model=target_model, **model_kwargs)  # assigns self.model, self.tokenizer, and self.model_name_or_path
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.use_prefix_cache = use_prefix_cache
        
        # === 记录自适应攻击参数 ===
        self.p_toxic_dict = p_toxic_dict
        self.adaptive_alpha = adaptive_alpha

        if use_prefix_cache and self.model.config.use_cache != use_prefix_cache:
            print(f"WARNING: setting model.config.use_cache={use_prefix_cache}")
            self.model.config.use_cache = use_prefix_cache
        
        with open(targets_path, 'r', encoding='utf-8') as file:
            self.behavior_id_to_target = json.load(file)
        
        ### Eval Vars ###
        self.eval_steps = eval_steps
        self.eval_with_check_refusal = eval_with_check_refusal
        self.check_refusal_min_loss = check_refusal_min_loss
        self.early_stopping = early_stopping
        self.early_stopping_min_loss = early_stopping_min_loss
        self.starting_search_batch_size = starting_search_batch_size

        ## Instruction template
        template = get_template(**target_model)['prompt']
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")
        
    def generate_test_cases_single_behavior(self, behavior_dict, verbose=False, **kwargs):
        """
        Generates test cases for a single behavior
        """

        # starting search_batch_size, will auto reduce batch_size later if go OOM (resets for every new behavior)
        self.search_batch_size = self.starting_search_batch_size if self.starting_search_batch_size else self.search_width

        # ========== Behavior and Target str ==========
        behavior = behavior_dict['Behavior']
        context_str = behavior_dict['ContextString']
        behavior_id = behavior_dict['BehaviorID']
        
        target = self.behavior_id_to_target[behavior_id]
        behavior += ' '
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        print(f'Behavior: {behavior_id} || Target: {target}')
        ### Targeted Model and Tokenier ###
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        ### GCG hyperparams ###
        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        allow_non_ascii = self.allow_non_ascii
        search_width = self.search_width
        before_tc = self.before_tc
        after_tc = self.after_tc

        ### Eval vars ###
        eval_steps = self.eval_steps
        eval_with_check_refusal = self.eval_with_check_refusal
        check_refusal_min_loss = self.check_refusal_min_loss
        early_stopping = self.early_stopping
        early_stopping_min_loss = self.early_stopping_min_loss

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[0]  # can be larger than tokenizer.vocab_size for some models
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        # remove redundant tokens
        if not_allowed_tokens is not None:
            not_allowed_tokens = torch.unique(not_allowed_tokens)

        # ========== setup optim_ids and cached components ========== #
        optim_ids = tokenizer(adv_string_init, return_tensors="pt", add_special_tokens=False).to(device)['input_ids']
        num_optim_tokens = len(optim_ids[0])

        cache_input_ids = tokenizer([before_tc], padding=False)['input_ids'] # some tokenizer have <s> for before_tc
        cache_input_ids += tokenizer([behavior, after_tc, target], padding=False, add_special_tokens=False)['input_ids']
        cache_input_ids = [torch.tensor(input_ids, device=device).unsqueeze(0) for input_ids in cache_input_ids] # make tensor separately because can't return_tensors='pt' in tokenizer
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [embed_layer(input_ids) for input_ids in cache_input_ids]
        
        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            input_embeds = torch.cat([before_embeds, behavior_embeds], dim=1)
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values

        # ========== run optimization ========== #
        all_losses = []
        all_test_cases = []
        for i in tqdm(range(num_steps)):
            # ========== compute coordinate token_gradient ========== #
            # create input
            optim_ids_onehot = torch.zeros((1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype)
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            # forward pass - 强制输出 hidden_states
            if self.use_prefix_cache:
                input_embeds = torch.cat([optim_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache, output_hidden_states=True)
            else:
                input_embeds = torch.cat([before_embeds, behavior_embeds, optim_embeds, after_embeds, target_embeds], dim=1)
                outputs = model(inputs_embeds=input_embeds, output_hidden_states=True)
            
            logits = outputs.logits

            # compute Original Target Loss
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = target_ids
            loss_fct = CrossEntropyLoss()
            loss_target = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # === 新增：计算自适应 Evasion Loss ===
            loss_evasion = torch.tensor(0.0, device=device)
            if self.p_toxic_dict is not None:
                for layer_idx, p_tox in self.p_toxic_dict.items():
                    hidden_states = outputs.hidden_states[layer_idx]
                    p_tox = p_tox.to(hidden_states.device).to(hidden_states.dtype)
                    # 投影到毒性子空间计算能量
                    toxic_projection = torch.matmul(hidden_states, p_tox) 
                    layer_evasion_loss = (toxic_projection ** 2).sum(dim=-1).mean()
                    loss_evasion += layer_evasion_loss
            
            # 综合 Loss：既要生成目标词（越狱），又要避开毒性空间（规避防御）
            loss = loss_target + self.adaptive_alpha * loss_evasion

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices = sample_control(optim_ids.squeeze(0), token_grad.squeeze(0), search_width, topk=256, temp=1, not_allowed_tokens=not_allowed_tokens)

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            count = 0
            if "llama-3" in self.model.config._name_or_path.lower():
                new_sampled_top_indices = list(sampled_top_indices)
            else:
                for j in range(len(sampled_top_indices_text)):
                    # tokenize again
                    tmp = tokenizer(sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False).to(device)['input_ids'][0]
                    # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                    if not torch.equal(tmp, sampled_top_indices[j]):
                        count += 1
                        continue
                    else:
                        new_sampled_top_indices.append(sampled_top_indices[j])
            
            if len(new_sampled_top_indices) == 0:
                print('All removals; defaulting to keeping all')
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)
            
            if count >= search_width // 2:
                print('\nLots of removals:', count)
            
            new_search_width = search_width - count

            # ========== Compute loss on these candidates and take the argmin. ========== #
            # create input
            sampled_top_embeds = embed_layer(sampled_top_indices)
            if self.use_prefix_cache:
                input_embeds = torch.cat([sampled_top_embeds,
                                        after_embeds.repeat(new_search_width, 1, 1),
                                        target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            else:
                input_embeds = torch.cat([before_embeds.repeat(new_search_width, 1, 1),
                                        behavior_embeds.repeat(new_search_width, 1, 1),
                                        sampled_top_embeds,
                                        after_embeds.repeat(new_search_width, 1, 1),
                                        target_embeds.repeat(new_search_width, 1, 1)], dim=1)
            
            # 这里 compute_candidates_loss 也会返回加上了自适应惩罚项的 loss
            loss_candidates = find_executable_batch_size(self.compute_candidates_loss, self.search_batch_size)(input_embeds, target_ids)

            # ========== Update the optim_ids with the best candidate ========== #
            optim_ids = sampled_top_indices[loss_candidates.argmin()].unsqueeze(0)
            
            test_case_ids = torch.cat([behavior_ids, optim_ids], dim=1)
            test_case = tokenizer.decode(test_case_ids[0])
            all_test_cases.append(test_case)
            current_loss = loss_candidates.min().item()
            all_losses.append(current_loss)

            # ========== Eval and Early Stopping ========== #
            if (i % eval_steps == 0) or (i == num_steps - 1):
                p_output = f'\n===>Step {i}\n===>Test Case: {test_case}\n===>Loss: {current_loss}'
                if eval_with_check_refusal and current_loss < check_refusal_min_loss:
                    input_str = self.template.format(instruction=test_case)
                    is_refusal, completions, _ = check_refusal_completions(model, tokenizer, inputs=[input_str])
                    p_output += f"\n\n===>Completion: {completions[0]}"
                    if not is_refusal[0]:
                        break

                if verbose:
                    print(p_output)

            if early_stopping and current_loss < early_stopping_min_loss:
                print(f'Early stopping at step {i} with loss {current_loss}')
                print()
                break

            del input_embeds, sampled_top_embeds, logits, shift_logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        logs = {'final_loss': current_loss, 'all_losses': all_losses, 'all_test_cases': all_test_cases}

        return test_case, logs
        
    def compute_candidates_loss(self, search_batch_size, input_embeds, target_ids):
        # 1. 导入必要的 Cache 工具类
        from transformers.cache_utils import DynamicCache

        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        all_loss = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.use_prefix_cache:
                    # --- 核心修复：重构 prefix_cache 结构 ---
                    prefix_cache_batch = []
                    for layer in self.prefix_cache:
                        layer_expanded = tuple(kv.expand(current_batch_size, -1, -1, -1) for kv in layer)
                        prefix_cache_batch.append(layer_expanded)
                    
                    past_key_values = DynamicCache.from_legacy_cache(tuple(prefix_cache_batch))
                    
                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=past_key_values, output_hidden_states=True)
                else:
                    outputs = self.model(inputs_embeds=input_embeds