import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# ==========================================
# 全局字体和样式设置（单栏）
# ==========================================
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 8})

# ==========================================
# 1. 准备数据并求平均
# ==========================================
labels = ['MLP', r'$\mathbf{W}_v$', r'$\mathbf{W}_o$(Rand)', r'$\mathbf{W}_o$(Ours)']

vicuna_asr  = [0.3098, 0.5885, 0.5449, 0.2070]
vicuna_util = [5.15, 5.31, 5.26, 5.19]

llama_asr   = [0.5118, 0.4704, 0.3969, 0.1808]
llama_util  = [6.19, 6.15, 5.75, 6.44]

# --- 修正 Mistral 数据 ---
# 之前错误数据: mistral_asr = [0.6415, 0.6085, 0.0003, 0.1105]; mistral_util = [6.29, 6.28, 1.00, 5.38]
# 现在使用重测的 wo (0.6628, 5.84) 更新第三项
mistral_asr  = [0.6415, 0.6085, 0.6628, 0.1105]
mistral_util = [6.29, 6.28, 5.84, 5.38]

# 计算三个模型的平均值
avg_asr = np.mean([vicuna_asr, llama_asr, mistral_asr], axis=0)
avg_util = np.mean([vicuna_util, llama_util, mistral_util], axis=0)

# ==========================================
# 2. 创建图表
# ==========================================
fig, ax1 = plt.subplots(figsize=(3.4, 3.0))

x = np.arange(len(labels))
width = 0.35

color_asr  = '#9fc5e8'  # 浅蓝
color_util = '#cc0000'  # 红色

# 绘制 ASR 柱状图（左轴）
rects1 = ax1.bar(x - width/2, avg_asr, width, label='Avg. ASR (Left)', 
                 color=color_asr, edgecolor='black', linewidth=0.8, zorder=3)

# 绘制 Utility 柱状图（右轴）
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, avg_util, width, label='Avg. Utility (Right)', 
                 color=color_util, edgecolor='black', linewidth=0.8, alpha=0.8, zorder=3)

# ==========================================
# 3. 坐标轴和标签设置
# ==========================================
# 左轴 (ASR)
ax1.set_ylabel(r'Average ASR $\downarrow$', fontsize=9, fontweight='bold')
ax1.set_ylim(0.0, 0.65) # 【修改点】：因为修正后的 wo 均值达到了 0.5389，Y轴上限需要从 0.60 提到 0.65 防止顶头
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)
ax1.tick_params(axis='y', labelsize=8)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 右轴 (Utility)
ax2.set_ylabel(r'Average Utility $\uparrow$', color=color_util, fontsize=9, fontweight='bold')
ax2.set_ylim(0.0, 7.5) 
ax2.tick_params(axis='y', labelcolor=color_util, labelsize=8)

# 加粗边框
for spine in ax1.spines.values():
    spine.set_linewidth(1.0)
for spine in ax2.spines.values():
    spine.set_linewidth(1.0)

# ==========================================
# 4. 数值标注（标在柱子上方）
# ==========================================
# ASR 标注
for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f'{height:.3f}',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 2),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=7, color='black')

# Utility 标注
for rect in rects2:
    height = rect.get_height()
    ax2.annotate(f'{height:.2f}',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 2),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=7, color=color_util, fontweight='bold')

# ==========================================
# 5. 图例排版
# ==========================================
handles = [rects1, rects2]
fig.legend(handles, ['Avg. ASR (Left axis)', 'Avg. Utility (Right axis)'], 
           loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=2, fancybox=True, shadow=False, fontsize=8,
           facecolor='white', framealpha=1.0)

# ==========================================
# 6. 输出保存
# ==========================================
fig.tight_layout()
plt.savefig("avg_intervention_comparison_singlecol.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()