import json
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
def plot_academic_joint_comparison(timestamp):
    # ==========================================
    # 1. 全局样式配置 (保持学术风格)
    # ==========================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    plt.rcParams['pdf.fonttype'] = 42  # 关键：使用TrueType字体
    plt.rcParams['ps.fonttype'] = 42
    print(f"PDF字体类型: {plt.rcParams['pdf.fonttype']}") 

    # 方法配置
    modes = ["pure_naive", "cagrad"]
    labels = {"pure_naive": "Standard Optimization", "cagrad": "Ours (CAGrad)"}
    
    # --- 关键修改：为不同方法分配不同颜色 ---
    # Standard 用灰色，CAGrad 用红色，形成鲜明对比
    method_colors = {"pure_naive": "#7F8C8D", "cagrad": "#D62728"}
    styles = {"pure_naive": "--", "cagrad": "-"}
    
    # 创建 2x1 布局
    fig, axes = plt.subplots(2, 1,figsize=(6, 8))
    
    # 读取数据
    all_data = {}
    for mode in modes:
        file_name = f"history_{mode}_{timestamp}.json"
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                all_data[mode] = json.load(f)

    if not all_data:
        print("未找到数据文件")
        return

    # ==========================================
    # 子图 1: 擦除性能对比 (Loss Tox)
    # ==========================================
    ax_top = axes[0]
    for mode in modes:
        if mode in all_data:
            ax_top.plot(all_data[mode]['step'], all_data[mode]['loss_clean'], 
                        color=method_colors[mode], linewidth=2.5, 
                        linestyle=styles[mode], label=labels[mode])
    
    ax_top.set_ylabel(r'Loss', fontweight='bold')
    ax_top.set_title(r"(a)Harmful Signal Suppression ($\mathcal{L}_{tox}$)", fontweight='bold', pad=15)
    ax_top.set_xlabel('Optimization Steps', fontweight='bold')
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_top.legend(loc='upper right', frameon=True)
    ax_top.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # 只显示4个刻度

    # ==========================================
    # 子图 2: 注入性能对比 (Loss Ref)
    # ==========================================
    ax_bottom = axes[1]
    for mode in modes:
        if mode in all_data:
            ax_bottom.plot(all_data[mode]['step'], all_data[mode]['loss_align'], 
                           color=method_colors[mode], linewidth=2.5, 
                           linestyle=styles[mode], label=labels[mode])
    
    ax_bottom.set_ylabel(r'Loss', fontweight='bold')
    ax_bottom.set_title(r"(b)Refusal Logic Injection ($\mathcal{L}_{ref}$)", fontweight='bold', pad=15)
    ax_bottom.set_xlabel('Optimization Steps', fontweight='bold')
    ax_bottom.grid(True, linestyle='--', alpha=0.5)
    ax_bottom.legend(loc='upper right', frameon=True)
    ax_bottom.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.tight_layout()
    
    save_path = f"head_to_head_diff_color_{timestamp}"
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    print(f"✨ 对比图已生成 (同图异色版): {save_path}")
    plt.show()

plot_academic_joint_comparison(1776433794)

