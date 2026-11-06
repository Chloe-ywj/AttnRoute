import matplotlib.pyplot as plt

# 提取的数据
epochs = [0.5, 0.6, 0.7, 0.8, 0.9]  # 假设的 x 轴（如 Epoch 或 Step）
gcg = [0.1061, 0.0576, 0.1182, 0.1030, 0.0788]
autodan = [0.1303, 0.0485, 0.1606, 0.0606, 0.0455]
pair = [0.0788, 0.0364, 0.1515, 0.1091, 0.0515]

# 全局字体设置：学术论文常使用 Serif 字体 (如 Times New Roman)
plt.rcParams['font.family'] = 'serif'

# 创建画布，设置较高的分辨率(dpi=300)以满足期刊出版要求
plt.figure(figsize=(8, 5), dpi=300)

# 绘制折线图，使用不同的颜色、线型和标记以增强区分度
plt.plot(epochs, gcg, marker='o', linestyle='-', color='#1f77b4', label='gcg', linewidth=2, markersize=7)
plt.plot(epochs, autodan, marker='s', linestyle='--', color='#ff7f0e', label='autodan', linewidth=2, markersize=7)
plt.plot(epochs, pair, marker='^', linestyle='-.', color='#2ca02c', label='pair', linewidth=2, markersize=7)

# 设置坐标轴标签和标题
# 您可以根据实际的实验内容修改 'Epoch' 和 'Metric Value'
plt.xlabel('$\eta$', fontsize=12, fontweight='bold')
plt.ylabel('ASR(%)', fontsize=12, fontweight='bold')

# 设置 x 轴的刻度只显示整数
plt.xticks(epochs)

# 添加图例，去掉边框使其看起来更清爽（或保留边框并设置透明度）
plt.legend(frameon=True, loc='best', fontsize=11, edgecolor='black')

# 添加淡色网格线，帮助读取数据同时不喧宾夺主
plt.grid(True, linestyle=':', alpha=0.7)

# 调整布局，防止标签被截断
plt.tight_layout()

# 保存为高分辨率图片 (PDF 或 PNG 格式适合放入论文)
plt.savefig('ablation_study_chart.pdf', format='pdf')
# plt.savefig('ablation_study_chart.png', dpi=300)

# 显示图表
plt.show()

