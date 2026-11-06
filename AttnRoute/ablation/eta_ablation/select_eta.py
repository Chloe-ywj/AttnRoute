import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据准备
# ==========================================
etas = [0.5,0.6,0.7, 0.8, 0.9]
layers = list(range(7, 21))

# 这是模拟的分数矩阵（行对应 Layer，列对应 Eta）
# 实际跑完消融实验后，你可以把你终端里打印出的真实分数填进来
# 或者使用 pd.read_csv("your_results.csv") 来动态加载
scores = np.array([
    [1.8301, 1.8201, 1.8127, 1.8006, 1.7899],#7
    [1.3569, 1.3562, 1.3586, 1.3733, 1.3843],
    [1.4412, 1.4514, 1.4522, 1.4594, 1.4620],
    [1.3590, 1.3587, 1.3691, 1.3757, 1.3851], # Layer 10
    [1.2535, 1.2752, 1.2767, 1.2933, 1.3034], # Layer 11
    [1.4976, 1.4779, 1.4901, 1.4879, 1.4733], # Layer 12 (全局最高点)
    [1.6791, 1.6635, 1.6578, 1.6453, 1.6306], # Layer 13
    [1.5185, 1.5024, 1.5019, 1.5104, 1.5021], # Layer 14
    [1.4112, 1.4320, 1.4559, 1.4652, 1.4718], # Layer 15
    [1.4700, 1.4751, 1.4712, 1.4850, 1.4884], # Layer 16
    [1.1847, 1.1893, 1.2017, 1.2025, 1.2077], # Layer 17
    [1.3621, 1.3723, 1.3738, 1.3728, 1.3759], # Layer 18
    [1.4756, 1.4721, 1.4875, 1.5010, 1.4992], # Layer 19
    [1.6984, 1.6271, 1.6145, 1.5795, 1.5454]  # Layer 20
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
                      cbar_kws={'label': 'Separation Score (Signal-to-Noise Ratio)'},
                      linewidths=1, # 增加格子之间的白线间隔，更美观
                      linecolor='white',
                      ax=ax)

# ==========================================
# 4. 图表细节美化
# ==========================================
plt.title(r"Ablation on SVD Energy Threshold ($\eta$) and Layer Selection", fontsize=16, pad=20, fontweight='bold')
plt.xlabel(r"Energy Threshold ($\eta$)", fontsize=14, labelpad=10)
plt.ylabel("Model Layer", fontsize=14, labelpad=10)

# 把 Y 轴的数字摆正（默认可能会旋转90度）
plt.yticks(rotation=0)
# 把 X 轴的数字放在格子中间对齐
plt.xticks(ha='center')

# ==========================================
# 5. 学术杀手锏：高亮最优点 (Highlight Peak)
# ==========================================
# 找到最高分的坐标：Layer 12 在行索引是 2，Eta 0.90 在列索引也是 2
# 我们给这个格子加上一个显眼的红色边框
peak_row_idx = 2
peak_col_idx = 2
ax.add_patch(plt.Rectangle((peak_col_idx, peak_row_idx), 1, 1, 
                           fill=False, 
                           edgecolor='red', 
                           lw=4, 
                           clip_on=False))

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

