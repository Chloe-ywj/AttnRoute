import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
x = [1.0, 1.5, 2.0, 2.5]
x_pos = np.arange(len(x)) * 0.6

asr_data = [0.0717, 0.0616, 0.0545, 0.0515]
mt_bench_data = [5.35625, 5.41625, 5.4125, 5.4125]

# 2. 设置画布
plt.rcParams['font.family'] = 'serif'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=False)

# 使用原图的标准配色
color_blue = '#1F77B4'
color_red = '#D94747'
bar_width = 0.3

# ================= 上图：ASR (柱状图 + 折线图) =================
ax1.bar(x_pos, asr_data, width=bar_width, color=color_blue, alpha=0.4, zorder=1)
line1, = ax1.plot(x_pos, asr_data, color=color_blue, marker='D', markersize=5, 
                  linewidth=1.5, label='Avg. ASR', zorder=3)

# 增加数据标签 (ASR)
for i in range(len(x_pos)):
    # 将标签放在柱子顶部偏上一点的位置
    ax1.text(x_pos[i], asr_data[i] + 0.002, f'{asr_data[i]:.3f}', 
             ha='center', va='bottom', fontsize=9, color=color_blue, fontweight='normal')

ax1.set_ylabel('ASR ↓', color=color_blue, fontsize=9.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_blue)
ax1.set_ylim(0, 0.09) 
ax1.set_yticks(np.arange(0, 0.091, 0.02))

# x 轴设置：取消刻度线，紧贴
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x, fontsize=9)
ax1.tick_params(axis='x', length=0, pad=2)

# ================= 下图：MT-Bench (柱状图 + 折线图) =================
ax2.bar(x_pos, mt_bench_data, width=bar_width, color=color_red, alpha=0.4, zorder=1)
line2, = ax2.plot(x_pos, mt_bench_data, color=color_red, marker='D', markersize=5, 
                  linewidth=1.5, label='MT-Bench', zorder=3)

# 增加数据标签 (MT-Bench)
for i in range(len(x_pos)):
    # 将标签放在柱子顶部偏上一点的位置 (数值仅保留三位或四位小数，看起来更整洁)
    ax2.text(x_pos[i], mt_bench_data[i] + 0.0015, f'{mt_bench_data[i]:.3f}', 
             ha='center', va='bottom', fontsize=9, color=color_red, fontweight='normal')

ax2.set_ylabel('MT-Bench ↑', color=color_red, fontsize=9.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_red)
ax2.set_ylim(5.33, 5.435) 
ax2.set_yticks(np.arange(5.33, 5.431, 0.02))

ax2.set_xticks(x_pos)
ax2.set_xticklabels(x, fontsize=9)
ax2.tick_params(axis='x', length=0, pad=2)
ax2.set_xlabel('Dynamic Gain Coefficient ($\\alpha_{max}$)', fontsize=9, fontweight='bold', labelpad=3)

# ================= 通用美化 =================
for ax in [ax1, ax2]:
    ax.grid(False)
    ax.set_xlim(x_pos[0] - 0.4, x_pos[-1] + 0.4)

fig.tight_layout()
fig.subplots_adjust(hspace=0.15)

plt.savefig('alpha_ablation_dual_independent.pdf', format='pdf', bbox_inches='tight')
plt.show()