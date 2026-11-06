import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. 整理热力图原始数据 (假设你已经有了这个 DataFrame)
# ==========================================
# 行是 Layer (10-20)，列是 Eta (0.5-0.9)
data = {
    0.5: [1.39, 1.29, 1.58, 1.75, 1.55, 1.50, 1.55, 1.18, 1.39, 1.54, 1.88],
    0.6: [1.38, 1.30, 1.55, 1.73, 1.54, 1.49, 1.54, 1.18, 1.39, 1.53, 1.80],
    0.7: [1.38, 1.29, 1.55, 1.72, 1.53, 1.49, 1.53, 1.19, 1.39, 1.53, 1.76],
    0.8: [1.39, 1.30, 1.54, 1.71, 1.53, 1.49, 1.53, 1.19, 1.39, 1.53, 1.72],
    0.9: [1.39, 1.30, 1.52, 1.69, 1.53, 1.49, 1.53, 1.20, 1.39, 1.53, 1.68]
}
df = pd.DataFrame(data, index=range(10, 21))

# 计算每个 Eta 的统计量
etas = df.columns.astype(float)
means = df.mean()
stds = df.std()
maxs = df.max() # 找出每个 Eta 的最高分 (通常是 Layer 20)

# ==========================================
# 2. 绘制学术风折线图
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 5))

# 绘制平均分带误差棒 (代表整体水平)
ax.errorbar(etas, means, yerr=stds, fmt='-o', color='tab:blue', 
            linewidth=2, markersize=8, capsize=5, elinewidth=1, 
            label='Mean Score (across all layers)')

# 【视觉杀手锏】: 单独标记出每个 Eta 的最高分 (代表性能上限)
# 我们知道 Layer 20 总是最高，所以这一行代表了 Layer 20 的趋势
ax.plot(etas, maxs, '-*', color='tab:red', linewidth=1.5, markersize=12, 
        label='Max Score (Layer 20)')

# 高亮 eta=0.5 的区域
ax.axvspan(0.48, 0.52, color='gray', alpha=0.1)

# 设置
ax.set_xlabel(r'Threshold $\eta$', fontsize=12, fontweight='bold')
ax.set_ylabel('Separation Score', fontsize=12, fontweight='bold')
ax.set_xticks(etas)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11)

# 保存
plt.tight_layout()
plt.savefig("eta_ablation_line_plot.pdf", dpi=300)
print("✅ 折线图已生成: eta_ablation_line_plot.pdf")
plt.show()

