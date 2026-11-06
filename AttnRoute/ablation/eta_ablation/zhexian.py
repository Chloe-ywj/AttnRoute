import matplotlib.pyplot as plt

# 你的消融实验数据
eta = [0.5, 0.6, 0.7, 0.8, 0.9]
# 按照 my-test-model 0到4 的顺序填入 Average Score
scores = [4.556, 3.244, 3.913, 4.369, 3.394] 

plt.figure(figsize=(8, 5))

# 1. 画基础折线
plt.plot(eta, scores, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

# 2. 高亮 eta=0.5 的最佳结果
plt.plot(eta[0], scores[0], marker='*', color='red', markersize=16, label='Optimal $\\eta=0.5$')

# 3. 为每个点添加数值标签
for i, (x, y) in enumerate(zip(eta, scores)):
    # 稍微调整文本位置，避免遮挡折线
    offset = 0.05 if i != 1 else -0.12 
    plt.text(x, y + offset, f'{y:.3f}', ha='center', va='bottom' if offset > 0 else 'top', fontsize=11, fontweight='bold')

# 4. 设置坐标轴和标题
plt.xlabel('$\\eta$', fontsize=12)
plt.ylabel('MT-Bench Average Score', fontsize=12)
plt.xticks(eta)

# 5. 美化网格、图例和Y轴范围
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.ylim(3.0, 5.0)

# 6. 保存高质量图片
plt.savefig('eta_ablation_plot.pdf', dpi=300, bbox_inches='tight') # 写论文建议保存为 pdf 矢量图
plt.show()

