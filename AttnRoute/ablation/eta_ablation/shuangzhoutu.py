import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 填入真实消融数据 (精确到两位小数)
# ==========================================
etas = [0.5, 0.6, 0.7, 0.8, 0.9]

# 根据真实表格计算出的 ASR 平均值 (%)
asr_avg = [7.07, 8.89, 7.17, 15.15, 18.79]

# MT-Bench 真实分数
mt_bench = [5.381, 5.5, 5.024, 5.811, 5.861]
# ==========================================
# 2. 全局样式配置 (学术风格)
# ==========================================
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots(figsize=(6, 5))

# ==========================================
# 3. 绘制左轴 (ASR: 越低越好，使用警示色红色)
# ==========================================
color_asr = 'tab:red'
ax1.set_xlabel(r'Threshold $\eta$', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Average ASR (%)', color=color_asr, fontsize=12, fontweight='bold')
line1 = ax1.plot(etas, asr_avg, marker='o', markersize=8, color=color_asr, 
                 linewidth=2.5, linestyle='-', label='Average ASR')
ax1.tick_params(axis='y', labelcolor=color_asr, labelsize=12)
ax1.set_xticks(etas)
ax1.grid(True, linestyle='--', alpha=0.6) 

# 设置左轴的上下限，让折线更居中，留出顶部空间给图例
ax1.set_ylim(2, 20)

# ==========================================
# 4. 绘制右轴 (MT-Bench: 越高越好，使用安全色蓝色)
# ==========================================
ax2 = ax1.twinx()  
color_mt = 'tab:blue'
ax2.set_ylabel('MT-Bench Score', color=color_mt, fontsize=12, fontweight='bold')
line2 = ax2.plot(etas, mt_bench, marker='s', markersize=8, color=color_mt, 
                 linewidth=2.5, linestyle='--', label='MT-Bench')
ax2.tick_params(axis='y', labelcolor=color_mt, labelsize=12)

# 设置右轴的上下限
ax2.set_ylim(4.8, 6.0)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', frameon=True, fontsize=11, 
           fancybox=True, shadow=False, edgecolor='gray')


plt.tight_layout()

plt.savefig("tradeoff_dual_axis_eta_exact.pdf", dpi=300, bbox_inches='tight')
plt.savefig("tradeoff_dual_axis_eta_exact.png", dpi=300, bbox_inches='tight')

print("✅ 包含精确数据的双轴折线图已生成！请查看 tradeoff_dual_axis_eta_exact.pdf")
plt.show()

