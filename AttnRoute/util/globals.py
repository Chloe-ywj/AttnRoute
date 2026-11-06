from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()

# 1. 数据集目录 (必须)
# run_attnroute_visualization.py 中: ds = ds_class(DATA_DIR, ...)
DATA_DIR = ROOT_DIR / "data"

# 2. 超参数目录 (必须)
# run_attnroute_visualization.py 中: params_path = HPARAMS_DIR / hparams_fname
HPARAMS_DIR = ROOT_DIR / "hparams"

# 3. 结果输出目录 (必须)
# run_attnroute_visualization.py 中: run_dir = RESULTS_DIR / out_name
RESULTS_DIR = ROOT_DIR / "results"

#4. 统计数据目录 (必须 - 用于 Stage 4 权重更新)
# attnroute_main.py 调用 get_cov() 时，底层通常需要此路径查找协方差矩阵
STATS_DIR = DATA_DIR / "stats1"

# Remote URLs
REMOTE_ROOT_URL = "https://memit.baulab.info"

