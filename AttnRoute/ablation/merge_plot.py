import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as patches

# ==========================================
# 1. 数据准备 (基于原始热力图数据)
# ==========================================
data = {
    0.5: [1.39, 1.29, 1.58, 1.75, 1.55, 1.50, 1.55, 1.18, 1.39, 1.54, 1.88],
    0.6: [1.38, 1.30, 1.55, 1.73, 1.54, 1.49, 1.54, 1.18, 1.39, 1.53, 1.80],
    0.7: [1.38, 1.29, 1.55, 1.72, 1.53, 1.49, 1.53, 1.19, 1.39, 1.53, 1.76],
    0.8: [1.39, 1.30, 1.54, 1.71, 1.53, 1.49, 1.53, 1.19, 1.39, 1.53, 1.72],
    0.9: [1.39, 1.30, 1.52, 1.69, 1.53, 1.49, 1.53, 1.20, 1.39, 1.53, 1.68]
}
layers = list(range(10, 21)) # Y轴标签是10-20
df = pd.DataFrame(data, index=layers)

# ==========================================
# 2. 全局样式设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
fig = plt.figure(figsize=(12, 6))

# ------------------------------------------
# 子图 (a): 热力图 - 精准方框高亮层优势
# ------------------------------------------
ax1 = plt.subplot(121)
# 绘制基础热力图
heatmap=sns.heatmap(df, annot=True, fmt=".2f", cmap='Blues', linewidths=.5, 
            cbar_kws={'label': 'Separation Score'}, ax=ax1)
cbar = heatmap.collections[0].colorbar
cbar.set_label(cbar.ax.get_ylabel(), weight='bold', fontsize=12)
ax1.set_title('(a) Robust Harmful-Layer Location', fontweight='bold', pad=15)
ax1.set_xlabel(r'Threshold $\eta$', fontweight='bold', fontsize=12)
ax1.set_ylabel('Model Layer', fontweight='bold', fontsize=12)
ax1.set_yticklabels(layers, rotation=0, ha='right', va='center')

# 💡【核心更新：方框逻辑】
# patches.Rectangle((x_index, y_index), width, height, ...)
# 注意：heatmap的x和y都是从0开始的索引。
# 目标：高亮 Layer 13 (Y轴索引为3) 和 Layer 20 (Y轴索引为10)。横向跨越所有eta (宽度=5)。

rect1 = patches.Rectangle((0, 3), 5, 1, fill=False, edgecolor='tab:red', lw=2.5, linestyle='-', zorder=5,clip_on=False)
ax1.add_patch(rect1)

# 方框 2: 高亮第 20 层 (绝对最高分层)
rect2 = patches.Rectangle((0, 10), 5, 1, fill=False, edgecolor='tab:red', lw=2.5, linestyle='-', zorder=5,clip_on=False)
ax1.add_patch(rect2)
# ------------------------------------------
# 子图 (b): 带误差棒折线图 - 保持原样 (统计论证 η=0.5 的优势)
# ------------------------------------------
ax2 = plt.subplot(122)
etas = df.columns.astype(float)
means = df.mean()
stds = df.std()
maxs = df.max()  # 每个 η 下的最佳层表现

# 绘制平均分（带误差棒，显示整体表现更佳）
ax2.errorbar(etas, means, yerr=stds, fmt='-o', color='tab:blue', 
            linewidth=2, markersize=8, capsize=6, label='Mean Score (across all layers)', 
            markerfacecolor='white', markeredgewidth=2)

# 绘制最高分（星号，显示 η=0.5 能触及的最高防御潜力）
ax2.plot(etas, maxs, '--*', color='tab:red', linewidth=1.5, markersize=12, 
         label='Peak Score (Layer 20)')
ax2.axvspan(0.48, 0.52, color='gray', alpha=0.1)

ax2.set_title(r'(b) Statistical Superiority of $\eta=0.5$', fontweight='bold', pad=15)
ax2.set_xlabel(r'Threshold $\eta$', fontweight='bold', fontsize=12)
ax2.set_ylabel('Separation Score', fontweight='bold', fontsize=12)
ax2.set_xticks(etas)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(frameon=True, facecolor='white', framealpha=1)

plt.tight_layout()
plt.savefig("ablation_with_focused_patches.pdf", dpi=300, bbox_inches='tight')
plt.show()

