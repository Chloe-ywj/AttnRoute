import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据准备
# ==========================================
etas = [0.5, 0.6, 0.7, 0.8, 0.9]
layers = list(range(10, 21))

# 这是模拟的分数矩阵（行对应 Layer，列对应 Eta）
# 实际跑完消融实验后，你可以把你终端里打印出的真实分数填进来
# 或者使用 pd.read_csv("your_results.csv") 来动态加载
scores = np.array([
    [1.3892, 1.3835, 1.3847, 1.3855, 1.3884], # Layer 10
    [1.2939, 1.2964, 1.2919, 1.2969, 1.3004], # Layer 11
    [1.5778, 1.5523, 1.5469, 1.5371, 1.5229], # Layer 12 (全局最高点)
    [1.7496, 1.7304, 1.7187, 1.7060, 1.6908], # Layer 13
    [1.5501, 1.5384, 1.5327, 1.5328, 1.5263], # Layer 14
    [1.4967, 1.4911, 1.4940, 1.4933, 1.4920], # Layer 15
    [1.5535, 1.5441, 1.5318, 1.5315, 1.5280], # Layer 16
    [1.1767, 1.1810, 1.1892, 1.1920, 1.1964], # Layer 17
    [1.3911, 1.3948, 1.3946, 1.3935, 1.3942], # Layer 18
    [1.5416, 1.5294, 1.5298, 1.5319, 1.5279], # Layer 19
    [1.8769, 1.7990, 1.7650, 1.7208, 1.6796]  # Layer 20
])

# 转换为 DataFrame 格式，方便 Seaborn 处理
df = pd.DataFrame(scores, index=layers, columns=etas)

# ==========================================
# 2. 全局样式与字体配置
# ==========================================
# 论文中通常需要清晰、高对比度的样式
sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(10, 8)) # 设置图片比例

# ==========================================
# 3. 绘制热力图核心逻辑
# ==========================================
# cmap="YlGnBu" 是论文里最经典的配色（黄-绿-蓝），颜色越深代表值越大
# annot=True 会在每个格子里写上具体的分数
heatmap = sns.heatmap(df, 
                      annot=True, 
                      fmt=".2f", 
                      cmap="Blues", # 你也可以换成 "YlGnBu" 或 "rocket_r"
                      cbar_kws={'label': 'Separation Score'},
                      linewidths=1, # 增加格子之间的白线间隔，更美观
                      linecolor='white',
                      ax=ax)
plt.xlabel(r"Threshold ($\eta$)", fontsize=12, labelpad=10)
plt.ylabel("Model Layer", fontsize=12, labelpad=10)

# 把 Y 轴的数字摆正（默认可能会旋转90度）
plt.yticks(rotation=0)
# 把 X 轴的数字放在格子中间对齐
plt.xticks(ha='center')

# ==========================================
# 6. 保存与展示
# ==========================================
plt.tight_layout()

# 极其重要：写论文时，务必保存为 .pdf 矢量图格式，这样放大多少倍都不会糊！
plt.savefig("ablation_heatmap_eta.pdf", dpi=300, bbox_inches='tight')
# 同时存一张 png 方便在报告/PPT里预览
plt.savefig("ablation_heatmap_eta.png", dpi=300, bbox_inches='tight')

print("✅ 热力图已成功生成并保存为 PDF 和 PNG 格式。")
plt.show()

