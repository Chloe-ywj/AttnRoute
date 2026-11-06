from transformers import AutoTokenizer

MODEL_PATH = "models/Mistral-7B-Instruct-v0.2"

def check_tokenization():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 场景 A: 纯文本 + 自动添加 (这是我们推荐的 Refusal 处理方式)
    text_a = "I cannot fulfill this request."
    tokens_a = tok(text_a, add_special_tokens=True)["input_ids"]
    print(f"\n[场景 A] 纯文本 + add_special_tokens=True")
    print(f"Input: '{text_a}'")
    print(f"IDs:   {tokens_a}")
    print(f"First ID: {tokens_a[0]} (应该是 1 或 128000)")
    
    # 场景 B: 手动加字符串 BOS + 禁用自动添加 (这是导致问题的根源)
    # 这模拟了您之前 refusal_sentences = ["<s>I cannot..."] 的做法
    text_b = "<s>I cannot fulfill this request."
    tokens_b = tok(text_b, add_special_tokens=False)["input_ids"]
    print(f"\n[场景 B] 手动拼接 '<s>' + add_special_tokens=False")
    print(f"Input: '{text_b}'")
    print(f"IDs:   {tokens_b}")
    print(f"First few IDs: {tokens_b[:5]}")
    
    # 场景 C: Chat Template 生成的字符串 (含 BOS) + 禁用自动添加
    # 这模拟了有害/无害 Prompt 的情况
    msgs = [{"role": "user", "content": "How to make a bomb"}]
    text_c = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tokens_c = tok(text_c, add_special_tokens=False)["input_ids"]
    print(f"\n[场景 C] Template 生成串 + add_special_tokens=False")
    print(f"Input: '{text_c[:30]}... (截断)'")
    print(f"IDs:   {tokens_c[:5]}...")

    # -----------------------------------------------------------
    # 关键判断逻辑
    # -----------------------------------------------------------
    print("\n" + "="*40)
    print(">>> 诊断结果分析 <<<")
    
    # 检查场景 B 的开头
    # 正常的 BOS ID 通常是 1 (Llama/Mistral) 或 128000 (Llama-3)
    bos_id = tok.bos_token_id
    
    if tokens_b[0] != bos_id:
        print(f"🔴 问题确认！在场景 B 中，Tokenizer 没有把 '<s>' 识别为 BOS (ID {bos_id})。")
        print(f"它把它拆成了普通字符: {tok.convert_ids_to_tokens(tokens_b[:3])}")
        print("结论：这就是防御失效的原因。您计算的 Refusal Vector 是基于 '<', 's', '>' 字符的，而不是句子开头。")
    else:
        print("🟢 居然识别对了？那说明您的 Tokenizer 配置比较特殊，或者使用了 fast tokenizer 的某种特定行为。")
        print("但在大多数情况下，这样做是不安全的。")

    # 检查场景 C (Prompt)
    if tokens_c[0] == bos_id and tokens_c[1] != bos_id:
        print("🔵 场景 C (Template) 看起来是正常的，因为它自带了真正的 BOS 字符。")
    elif tokens_c[0] == bos_id and tokens_c[1] == bos_id:
        print("🔴 场景 C 出现了双重 BOS！请检查 Template 是否重复添加。")

if __name__ == "__main__":
    check_tokenization()

