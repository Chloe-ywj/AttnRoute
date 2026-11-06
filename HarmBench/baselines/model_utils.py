import os
import torch
import random
import json
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM
from huggingface_hub import login as hf_login
import ray
from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template
from inspect import signature
from peft import PeftModel

ALPACA_PROMPT = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
}
VICUNA_1_0_PROMPT = {
    "description": "Template used by Vicuna 1.0 and stable vicuna.",
    "prompt": "### Human: {instruction}\n### Assistant:",
}

VICUNA_PROMPT = {
    "description": "Template used by Vicuna.",
    "prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:",
}

OASST_PROMPT = {
    "description": "Template used by Open Assistant",
    "prompt": "<|prompter|>{instruction}<|endoftext|><|assistant|>"
}
OASST_PROMPT_v1_1 = {
    "description": "Template used by newer Open Assistant models",
    "prompt": "<|prompter|>{instruction}</s><|assistant|>"
}


LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
LLAMA2_CHAT_PROMPT = {
    "description": "Template used by Llama2 Chat",
    # "prompt": "[INST] {instruction} [/INST] "
    "prompt": "[INST] <<SYS>>\n"+LLAMA2_DEFAULT_SYSTEM_PROMPT+"\n<</SYS>>\n\n{instruction} [/INST] "
}

INTERNLM_PROMPT = { # https://github.com/InternLM/InternLM/blob/main/tools/alpaca_tokenizer.py
    "description": "Template used by INTERNLM-chat",
    "prompt": "<|User|>:{instruction}<eoh><|Bot|>:"
}

KOALA_PROMPT = { #https://github.com/young-geng/EasyLM/blob/main/docs/koala.md#koala-chatbot-prompts
    "description": "Template used by EasyLM/Koala",
    "prompt": "BEGINNING OF CONVERSATION: USER: {instruction} GPT:"
}

# Get from Rule-Following: cite
FALCON_PROMPT = { # https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1#6475a107e9b57ce0caa131cd
    "description": "Template used by Falcon Instruct",
    "prompt": "User: {instruction}\nAssistant:",
}

MPT_PROMPT = { # https://huggingface.co/TheBloke/mpt-30B-chat-GGML
    "description": "Template used by MPT",
    "prompt": '''<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|><|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n''',
}

DOLLY_PROMPT = {
    "description": "Template used by Dolly", 
    "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
}


OPENAI_CHATML_PROMPT = {
    "description": "Template used by OpenAI chatml", #https://github.com/openai/openai-python/blob/main/chatml.md
    "prompt": '''<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
'''
}

LLAMA2_70B_OASST_CHATML_PROMPT = {
    "description": "Template used by OpenAI chatml", #https://github.com/openai/openai-python/blob/main/chatml.md
    "prompt": '''<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
'''
}

FALCON_INSTRUCT_PROMPT = { # https://huggingface.co/tiiuae/falcon-40b-instruct/discussions/1#6475a107e9b57ce0caa131cd
    "description": "Template used by Falcon Instruct",
    "prompt": "User: {instruction}\nAssistant:",
}

FALCON_CHAT_PROMPT = { # https://huggingface.co/blog/falcon-180b#prompt-format
    "description": "Template used by Falcon Chat",
    "prompt": "User: {instruction}\nFalcon:",
}

ORCA_2_PROMPT = {
    "description": "Template used by microsoft/Orca-2-13b",
    "prompt": "<|im_start|>system\nYou are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant"
}

MISTRAL_PROMPT = {
    "description": "Template used by Mistral Instruct",
    "prompt": "[INST] {instruction} [/INST]"
}

BAICHUAN_CHAT_PROMPT = {
    "description": "Template used by Baichuan2-chat",
    "prompt": "<reserved_106>{instruction}<reserved_107>"
}

QWEN_CHAT_PROMPT = {
    "description": "Template used by Qwen-chat models",
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
}

ZEPHYR_ROBUST_PROMPT = {
    "description": "",
    "prompt": "<|user|>\n{instruction}</s>\n<|assistant|>\n"
}

MIXTRAL_PROMPT = {
    "description": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "prompt": "[INST] {instruction} [/INST]"
}
########## CHAT TEMPLATE ###########

def get_template(model_name_or_path=None, chat_template=None, fschat_template=None, system_message=None, return_fschat_conv=False, **kwargs):
    print(f"   chat_template: {chat_template}")
    # ==== First check for fschat template ====
    if fschat_template or return_fschat_conv:
        fschat_conv = _get_fschat_conv(model_name_or_path, fschat_template, system_message)
        if return_fschat_conv: 
            print("Found FastChat conv template for", model_name_or_path)
            print(fschat_conv.dict())
            return fschat_conv
        else:
            fschat_conv.append_message(fschat_conv.roles[0], "{instruction}")
            fschat_conv.append_message(fschat_conv.roles[1], None) 
            TEMPLATE = {"description": f"fschat template {fschat_conv.name}", "prompt": fschat_conv.get_prompt()}
    # ===== Check for some older chat model templates ====
    elif chat_template == "wizard":
        TEMPLATE = VICUNA_PROMPT
    elif chat_template == "vicuna":
        TEMPLATE = VICUNA_PROMPT
    elif chat_template == "oasst":
        TEMPLATE = OASST_PROMPT
    elif chat_template == "oasst_v1_1":
        TEMPLATE = OASST_PROMPT_v1_1
    elif chat_template == "llama-2":
        TEMPLATE = LLAMA2_CHAT_PROMPT
    elif chat_template == "falcon_instruct": #falcon 7b / 40b instruct
        TEMPLATE = FALCON_INSTRUCT_PROMPT
    elif chat_template == "falcon_chat": #falcon 180B_chat
        TEMPLATE = FALCON_CHAT_PROMPT
    elif chat_template == "mpt":
        TEMPLATE = MPT_PROMPT
    elif chat_template == "koala":
        TEMPLATE = KOALA_PROMPT
    elif chat_template == "dolly":
        TEMPLATE = DOLLY_PROMPT
    elif chat_template == "internlm":
        TEMPLATE = INTERNLM_PROMPT
    elif chat_template == "mistral" or chat_template == "mixtral":
        TEMPLATE = MISTRAL_PROMPT
    elif chat_template == "orca-2":
        TEMPLATE = ORCA_2_PROMPT
    elif chat_template == "baichuan2":
        TEMPLATE = BAICHUAN_CHAT_PROMPT
    elif chat_template == "qwen":
        TEMPLATE = QWEN_CHAT_PROMPT
    elif chat_template == "zephyr_7b_robust":
        TEMPLATE = ZEPHYR_ROBUST_PROMPT
    else:
        # ======== Else default to tokenizer.apply_chat_template =======
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            template = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': '{instruction}'}] if system_message else [{'role': 'user', 'content': '{instruction}'}]
            prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
            # Check if the prompt starts with the BOS token
            # removed <s> if it exist (LlamaTokenizer class usually have this) as our baselines will add these if needed later
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt.replace(tokenizer.bos_token, "")
            TEMPLATE = {'description': f"Template used by {model_name_or_path} (tokenizer.apply_chat_template)", 'prompt': prompt}
        except:    
            assert TEMPLATE, f"Can't find instruction template for {model_name_or_path}, and apply_chat_template failed."

    print("Found Instruction template for", model_name_or_path)
    print(TEMPLATE)
        
    return TEMPLATE

def _get_fschat_conv(model_name_or_path=None, fschat_template=None, system_message=None, **kwargs):
    template_name = fschat_template
    if template_name is None:
        template_name = model_name_or_path
        print(f"WARNING: default to fschat_template={template_name} for model {model_name_or_path}")
        template = get_conversation_template(template_name)
    else:
        template = get_conv_template(template_name)
    
    # New Fschat version remove llama-2 system prompt: https://github.com/lm-sys/FastChat/blob/722ab0299fd10221fa4686267fe068a688bacd4c/fastchat/conversation.py#L1410
    if template.name == 'llama-2' and system_message is None:
        print("WARNING: using llama-2 template without safety system promp")
    
    if system_message:
        template.set_system_message(system_message)

    assert template and template.dict()['template_name'] != 'one_shot', f"Can't find fschat conversation template `{template_name}`. See https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py for supported template"
    
    return template


########## MODEL ###########

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": "auto"
}


def load_model_and_tokenizer(
    model_name_or_path,
    dtype='auto',
    device_map='auto',
    trust_remote_code=False,
    revision=None,
    token=None,
    num_gpus=1,
    ## tokenizer args
    use_fast_tokenizer=True,
    padding_side='left',
    legacy=False,
    pad_token=None,
    eos_token=None,
    ## dummy args passed from get_template()
    chat_template=None,
    fschat_template=None,
    system_message=None,
    return_fschat_conv=False,
    subfolder=None,
    **model_kwargs
):  
    if token:
        hf_login(token=token)
    model_kwargs = dict(model_kwargs)
    if model_kwargs.get("attn_implementation") == "flash_attention_2":
        try:
            import flash_attn
            print("✅ Flash Attention 2 已启用（加速模式）")
        except ImportError:
            print("⚠️ flash-attn 未安装，自动切换为 sdpa 模式（无需额外安装，速度依然很快）")
            model_kwargs["attn_implementation"] = "sdpa"   # 或者直接删掉这行参数也行
    if subfolder:
        print(f"📁 从子目录加载模型: {subfolder}")
        model_kwargs["subfolder"] = subfolder
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
        torch_dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype],
        device_map=device_map, 
        trust_remote_code=trust_remote_code, 
        revision=revision, 
        **model_kwargs).eval()

    # Init Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
        legacy=legacy,
        padding_side=padding_side,
    )
    if pad_token:
        tokenizer.pad_token = pad_token
    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Tokenizer.pad_token is None, setting to tokenizer.unk_token")
        tokenizer.pad_token = tokenizer.unk_token
        print("tokenizer.pad_token", tokenizer.pad_token)
    
    return model, tokenizer


def load_vllm_model(
    model_name_or_path,
    dtype='auto',
    trust_remote_code=False,
    download_dir=None,
    revision=None,
    token=None,
    quantization=None,
    num_gpus=1,
    use_fast_tokenizer=True,
    pad_token=None,
    eos_token=None,
    subfolder=None,
    **kwargs
):
    print(f"📌 num_gpus: {num_gpus}")
    if token:
        hf_login(token=token)

    if num_gpus > 1:
        _init_ray(reinit=False)

    lora_path = kwargs.pop("lora_path", None)
    
    # make it flexible if we want to add anything extra in yaml file
    model_kwargs = {k: kwargs[k] for k in kwargs if k in signature(LLM).parameters}

    if lora_path is not None:
        print(f"✅ vLLM 检测到 LoRA: {lora_path}")
        model_kwargs["enable_lora"] = True
        model_kwargs.setdefault("max_lora_rank", 32)
    if subfolder:
        print(f"📁 从子目录加载模型: {subfolder}")
        model_kwargs["subfolder"] = subfolder

    #                max_model_len=32768,
    model = LLM(model=model_name_or_path, 
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                download_dir=download_dir,
                revision=revision,
                quantization=quantization,
                tokenizer_mode="auto" if use_fast_tokenizer else "slow",
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.95,       # 把显存压榨到极限，给 KV cache 留足空间
                enforce_eager=True,
                max_model_len=32768,
                **model_kwargs)
    if pad_token:
        model.llm_engine.tokenizer.tokenizer.pad_token = pad_token
    if eos_token:
        model.llm_engine.tokenizer.tokenizer.eos_token = eos_token
    
    model.lora_path = lora_path

    return model
def _init_ray(num_cpus=8, reinit=False, resources={}):
    # --- 关键修改：将 global 声明移至此处 ---
    global ray  
    
    from transformers.dynamic_module_utils import init_hf_modules
    import ray
    import time
    import subprocess

    # 检查是否已经初始化
    if ('RAY_ADDRESS' in os.environ or ray.is_initialized()) and not reinit:
        print("Ray already initialized")
        return

    # 如果已经有 Ray 进程在运行，先清理
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True)
        time.sleep(2)
    except:
        pass

    # 设置 Ray 配置
    num_cpus = min([os.cpu_count(), num_cpus])
    os.environ['RAY_DEDUP_LOGS'] = '0'
    
    # 生成随机端口
    RAY_PORT = random.randint(6000, 6999)
    RAY_MIN_PORT = random.randint(10002, 20000)
    RAY_MAX_PORT = RAY_MIN_PORT + 99
    
    # 方法1：使用 ray.init() 直接启动（推荐）
    try:
        print(f"Starting Ray with {num_cpus} CPUs...")
        
        # 准备 runtime_env
        runtime_env = {
            "env_vars": {
                "RAY_DEDUP_LOGS": "0",
            }
        }
        
        # 准备资源
        if resources:
            os.environ['RAY_custom_unit_instance_resources'] = ",".join(resources.keys())
            ray_init_resources = resources
        else:
            ray_init_resources = None
        
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=num_cpus,
            resources=ray_init_resources,
            runtime_env=runtime_env,
            _temp_dir=f"/tmp/ray_{RAY_PORT}",
        )
        
        print(f"✅ Ray initialized successfully on port {RAY_PORT}")
        print(f"Ray cluster resources: {ray.cluster_resources()}")
        
    except Exception as e:
        print(f"❌ Ray init failed: {e}")
        print("Trying alternative method...")
        
        # 方法2：使用命令行启动
        try:
            resources_args = ""
            if resources:
                resources_args = f" --resources '{json.dumps(resources)}'"
            
            ray_start_command = [
                "ray", "start", "--head",
                f"--num-cpus={num_cpus}",
                f"--port={RAY_PORT}",
                f"--min-worker-port={RAY_MIN_PORT}",
                f"--max-worker-port={RAY_MAX_PORT}",
                "--disable-usage-stats",
                "--include-dashboard=False"
            ]
            
            if resources_args:
                ray_start_command.append(resources_args)
            
            print(f"Starting Ray with command: {' '.join(ray_start_command)}")
            result = subprocess.run(ray_start_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Ray start failed: {result.stderr}")
                return
            
            time.sleep(5)
            os.environ['RAY_ADDRESS'] = f"127.0.0.1:{RAY_PORT}"
            ray.init(address=f"127.0.0.1:{RAY_PORT}", ignore_reinit_error=True)
            print(f"✅ Ray connected successfully")
            
        except Exception as e2:
            print(f"❌ Both Ray initialization methods failed: {e2}")
            print("Continuing without Ray (single process mode)")
            # 这里之前报错是因为在函数末尾才声明 global
            import types
            fake_ray = types.ModuleType('ray')
            fake_ray.is_initialized = lambda: False
            fake_ray.init = lambda *a, **k: None
            fake_ray.cluster_resources = lambda: {}
            ray = fake_ray # 这里的赋值需要 global 声明在函数开头
            return

    # 初始化 HF 模块
    try:
        init_hf_modules()
    except:
        pass