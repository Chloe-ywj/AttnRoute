import json
import csv

# 从本地文件读取 JSON 数据
with open('/data/users/fanxin/yiwenjing/HarmBench/results/jailbreakbench/mistral/extracted_samples/pair/samples_converted.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 输出到 CSV 文件
with open('output3.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'prompt'])
    
    idx = 1
    for key, prompts in data.items():
        for prompt in prompts:
            writer.writerow([idx, prompt])
            idx += 1

print(f"转换完成！共生成 {idx-1} 条数据，已保存到 output.csv")