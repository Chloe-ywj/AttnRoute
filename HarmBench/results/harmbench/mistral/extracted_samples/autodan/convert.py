import json

def extract_goals_from_samples(input_file, output_file):
    """
    从samples_converted.json文件中提取每个key对应的完整值作为goal
    """
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化结果列表
    result = []
    id_counter = 0
    
    # 遍历每个key-value对
    for key, value in data.items():
        # value是一个列表，包含一个长文本
        if value and len(value) > 0:
            goal_content = value[0]  # 获取完整的文本内容
            
            # 添加到结果列表
            result.append({
                "id": id_counter,
                "goal": goal_content,
                "category": key
            })
            
            id_counter += 1
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"成功处理 {len(result)} 个条目")
    print(f"结果已保存到 {output_file}")
    print("\n第一个条目的预览：")
    if result:
        print(f"ID: {result[0]['id']}")
        print(f"Category: {result[0]['category']}")
        print(f"Goal (前100个字符): {result[0]['goal'][:100]}...")

# 使用示例
if __name__ == "__main__":
    input_file = "samples_converted.json"  # 输入文件名
    output_file = "output.json"  # 输出文件名
    extract_goals_from_samples(input_file, output_file)