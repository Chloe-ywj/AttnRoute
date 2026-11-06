#!/usr/bin/env python
# extract_jbb_targets.py

import csv
import json
import os
from pathlib import Path

def extract_jbb_targets(
    jbb_csv_path: str = './jbb_data/jbb_harmbench_format.csv',
    output_path: str = '../optimizer_targets/jbb_targets.json'
):
    """
    从 JBB CSV 文件中提取 target 映射为 JSON 文件
    
    Args:
        jbb_csv_path: JBB CSV 文件路径
        output_path: 输出 JSON 文件路径
    """
    print("="*60)
    print("🎯 从 JBB 提取 Target 映射")
    print("="*60)
    
    # 检查输入文件
    if not os.path.exists(jbb_csv_path):
        print(f"❌ 错误: 找不到 JBB CSV 文件")
        print(f"   路径: {jbb_csv_path}")
        print("\n请先运行保存 JBB 的脚本:")
        print("  python save_jbb_to_csv.py")
        return None
    
    print(f"📥 读取文件: {jbb_csv_path}")
    
    targets_dict = {}
    
    # 读取 CSV
    with open(jbb_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        print(f"📋 CSV 列名: {reader.fieldnames}")
        
        # 检查必要的列
        if 'BehaviorID' not in reader.fieldnames:
            print(f"❌ 错误: CSV 中没有 'BehaviorID' 列")
            return None
            
        if 'Target' not in reader.fieldnames:
            print(f"❌ 错误: CSV 中没有 'Target' 列")
            return None
        
        # 提取 target 映射
        row_count = 0
        for row in reader:
            behavior_id = row['BehaviorID']
            target = row['Target']
            
            if behavior_id and target:
                targets_dict[behavior_id] = target
                row_count += 1
                
                # 显示前3个示例
                if row_count <= 3:
                    print(f"\n📌 示例 {row_count}:")
                    print(f"  BehaviorID: {behavior_id}")
                    print(f"  Target: {target[:100]}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(targets_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 成功提取 {row_count} 个 target 映射")
    print(f"💾 已保存到: {output_path}")
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"  - 总条目数: {len(targets_dict)}")
    
    # 显示前5个 BehaviorID
    print(f"\n🔑 前5个 BehaviorID:")
    for i, key in enumerate(list(targets_dict.keys())[:5]):
        print(f"  {i+1}. {key}")
    
    # 验证文件内容
    print(f"\n🔍 验证 JSON 文件:")
    with open(output_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
        print(f"  - 成功加载 {len(loaded)} 个条目")
        print(f"  - 第一个 key: {list(loaded.keys())[0]}")
        print(f"  - 第一个 value: {loaded[list(loaded.keys())[0]][:50]}...")
    
    return targets_dict

def check_autodan_config():
    """
    检查 AutoDAN 配置文件中的 targets_path
    """
    config_path = './configs/method_configs/AutoDAN_config.yaml'
    
    if not os.path.exists(config_path):
        print(f"\n⚠️ 找不到 AutoDAN 配置文件: {config_path}")
        print("如果需要，请手动更新配置文件")
        return
    
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    current_targets = config.get('default_method_hyperparameters', {}).get('targets_path')
    
    print(f"\n📝 AutoDAN 配置检查:")
    print(f"  配置文件: {config_path}")
    print(f"  当前 targets_path: {current_targets}")
    
    expected_path = './data/optimizer_targets/jbb_targets.json'
    
    if current_targets != expected_path:
        print(f"  ⚠️ 建议更新为: {expected_path}")
        print(f"\n可以在配置文件中修改:")
        print("default_method_hyperparameters:")
        print("  targets_path: ./data/optimizer_targets/jbb_targets.json")

def main():
    """主函数"""
    # 从 JBB 提取 targets
    targets = extract_jbb_targets()

if __name__ == "__main__":
    main()