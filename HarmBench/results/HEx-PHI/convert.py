import json
import re
import os

def generate_slug(goal, index, length=40):
    """将 goal 转化为带序号的唯一 Key (如 distribute_pirated_movies_0)"""
    # 1. 清理字符串：只保留字母数字，空格转下划线
    slug = re.sub(r'[^a-zA-Z0-9\s]', '', goal).lower()
    slug = re.sub(r'\s+', '_', slug).strip('_')
    
    # 2. 截取长度并拼接索引，确保 330 条数据互不重合
    return f"{slug[:length]}_{index}"

def transform_autodan_to_hexphi(input_file, output_file):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"检测到目录不存在，已创建: {output_dir}")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    hex_phi_data = {}

    # 使用 enumerate 获取索引 i
    for i, entry in enumerate(raw_data):
        goal = entry.get("goal", "unknown_behavior")
        jailbreak_prompt = entry.get("jailbreak", "")
        
        # 传入索引 i，生成唯一的行为键名
        key = generate_slug(goal, i)
        
        # 强制每个 Key 对应一个独立的列表，从而产生 330 个类别
        hex_phi_data[key] = [jailbreak_prompt]

    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hex_phi_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成！")
    print(f"处理原始条数: {len(raw_data)}")
    print(f"生成独立类别数: {len(hex_phi_data)}")

# 运行路径
input_path = './llama/gcg/HEx-PHI-llama-3.json'
output_path = './llama/gcg/convert/HEx-PHI-llama-3.json'

transform_autodan_to_hexphi(input_path, output_path)
input_path = './llama/autodan/HEx-PHI-llama-3.json'
output_path = './llama/autodan/convert/HEx-PHI-llama-3.json'

transform_autodan_to_hexphi(input_path, output_path)
input_path = './llama/pair/HEx-PHI-llama-3.json'
output_path = './llama/pair/convert/HEx-PHI-llama-3.json'

transform_autodan_to_hexphi(input_path, output_path)