#!/usr/bin/env python
# save_jbb_to_csv.py

import jailbreakbench as jbb
import pandas as pd
from pathlib import Path

def save_jbb_behaviors_to_csv(output_path: str = "./jbb_behaviors.csv"):
    """
    将 JBB-Behaviors 原始数据集保存为 CSV
    
    Args:
        output_path: 输出 CSV 文件路径
    """
    print("📥 加载 JBB-Behaviors 数据集...")
    
    # 加载数据集
    dataset = jbb.read_dataset()  # 默认加载有害行为数据集
    
    print(f"✅ 数据集加载成功！")
    print(f"数据集包含 {len(dataset.goals)} 个行为")
    
    # 创建 DataFrame
    data = {
        'index': list(range(len(dataset.goals))),
        'behavior_id': dataset.behaviors,
        'goal': dataset.goals,
        'target': dataset.targets,
        'category': dataset.categories
    }
    
    df = pd.DataFrame(data)
    
    # 添加额外信息
    df['source'] = 'JBB-Behaviors'
    df['description'] = df['goal'].apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
    
    # 保存为 CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n💾 数据已保存到: {output_path}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 显示前几行
    print("\n📊 数据预览（前5行）:")
    print(df[['index', 'behavior_id', 'category', 'goal']].head())
    
    # 统计信息
    print("\n📈 类别分布:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    return df

def save_jbb_with_details(output_path: str = "./jbb_behaviors_detailed.csv"):
    """
    保存更详细的版本（包含原始字段说明）
    """
    dataset = jbb.read_dataset()
    
    # 创建详细 DataFrame
    data = []
    for i in range(len(dataset.goals)):
        data.append({
            'index': i,
            'behavior_id': dataset.behaviors[i],
            'goal': dataset.goals[i],
            'target': dataset.targets[i],
            'category': dataset.categories[i],
            'goal_length': len(dataset.goals[i]),
            'target_length': len(dataset.targets[i]),
            'has_target': bool(dataset.targets[i])
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n💾 详细数据已保存到: {output_path}")
    
    # 基本统计
    print(f"\n📊 统计信息:")
    print(f"平均 goal 长度: {df['goal_length'].mean():.1f} 字符")
    print(f"平均 target 长度: {df['target_length'].mean():.1f} 字符")
    print(f"最短 goal: {df['goal_length'].min()} 字符")
    print(f"最长 goal: {df['goal_length'].max()} 字符")
    
    return df

def save_jbb_as_harmbench_format(output_path: str = "./jbb_harmbench_format.csv"):
    """
    保存为 HarmBench 兼容格式
    """
    dataset = jbb.read_dataset()
    
    # HarmBench 格式
    data = {
        'BehaviorID': [f"jbb_{i}" for i in range(len(dataset.goals))],
        'Behavior': dataset.goals,
        'ContextString': [''] * len(dataset.goals),
        'FunctionalCategory': ['standard'] * len(dataset.goals),
        'BehaviorCategory': dataset.categories,
        'Target': dataset.targets,
        'Source': ['JBB-Behaviors'] * len(dataset.goals)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n💾 HarmBench 格式已保存到: {output_path}")
    print(f"数据形状: {df.shape}")
    print("\n列名:", list(df.columns))
    print("\n前3行预览:")
    print(df[['BehaviorID', 'BehaviorCategory', 'Behavior']].head(3))
    
    return df

if __name__ == "__main__":
    # 创建输出目录
    Path("./jbb_data").mkdir(exist_ok=True)
    
    # 1. 保存标准格式
    df1 = save_jbb_behaviors_to_csv("./jbb_data/jbb_behaviors.csv")
    
    # 2. 保存详细格式
    df2 = save_jbb_with_details("./jbb_data/jbb_behaviors_detailed.csv")
    
    # 3. 保存 HarmBench 兼容格式
    df3 = save_jbb_as_harmbench_format("./jbb_data/jbb_harmbench_format.csv")
    
    print("\n" + "="*60)
    print("✅ 所有格式保存完成！")
    print("文件列表:")
    print("  - ./jbb_data/jbb_behaviors.csv")
    print("  - ./jbb_data/jbb_behaviors_detailed.csv")
    print("  - ./jbb_data/jbb_harmbench_format.csv")