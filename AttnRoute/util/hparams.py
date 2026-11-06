import json
from dataclasses import dataclass

@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)
        
    def save_to_json(self, json_path):
        import json
        import os
        from typing import get_origin

        # 1. 构造一个纯净的字典
        raw_data = vars(self)
        serializable_data = {}
        
        for k, v in raw_data.items():
            # 排除私有属性、方法
            if k.startswith('__') or callable(v):
                continue
            
            # 关键修复：排除类型对象 (_GenericAlias) 和 类型本身
            if hasattr(v, '__origin__') or isinstance(v, type):
                continue
            
            # 基础类型直接保存
            if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                serializable_data[k] = v
            # 集合和元组转列表
            elif isinstance(v, (set, tuple)):
                serializable_data[k] = list(v)
            else:
                # 如果是其他对象，尝试看能不能转字符串，或者直接忽略
                try:
                    json.dumps(v) # 测试是否可序列化
                    serializable_data[k] = v
                except TypeError:
                    continue

        # 2. 安全写入逻辑：先写 .tmp 文件，防止因报错导致原文件变为空
        temp_path = str(json_path) + ".tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=4)
            # 只有写入成功才会替换原文件
            os.replace(temp_path, json_path)
            print(f"Hyperparameters successfully updated in {json_path}")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Failed to save JSON: {e}")
            raise e


