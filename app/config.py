import os

# ─── 语言设置 ───────────────────────────────────────────────
LANGUAGE = "zh"  # zh=中文  en=English
APP_VERSION = "v2.3.1"

# ─── 路径 ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# User data directory.
# Default keeps the current portable/development behavior. The installer
# launcher sets GUANGHUAN_STOCK_DATA_DIR to LocalAppData so upgrades do not
# overwrite watchlists, prediction history, logs, or self-learning models.
APP_DATA_DIR = os.environ.get("GUANGHUAN_STOCK_DATA_DIR") or BASE_DIR

OUTPUT_DIR = os.path.join(APP_DATA_DIR, "output")
CACHE_DIR = os.path.join(APP_DATA_DIR, ".cache")
HISTORY_DIR = os.path.join(APP_DATA_DIR, "history")
PREDICTION_LOG_DIR = os.path.join(APP_DATA_DIR, "prediction_logs")
PERFORMANCE_LOG_DIR = os.path.join(APP_DATA_DIR, "performance_logs")
ONLINE_MODEL_DIR = os.path.join(APP_DATA_DIR, "online_models")
SCHEDULER_DIR = os.path.join(APP_DATA_DIR, "scheduler")
BACKTEST_LOG_DIR = os.path.join(APP_DATA_DIR, "backtest_logs")
WATCHLIST_FILE = os.path.join(APP_DATA_DIR, "watchlist.json")

for _d in (OUTPUT_DIR, CACHE_DIR, HISTORY_DIR, PREDICTION_LOG_DIR, PERFORMANCE_LOG_DIR, ONLINE_MODEL_DIR, SCHEDULER_DIR, BACKTEST_LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── 定时分析 / 预警 ────────────────────────────────────────
ALERT_BUY_THRESHOLD  = 65   # 评分高于此值触发"买入"预警
ALERT_SELL_THRESHOLD = 35   # 评分低于此值触发"卖出"预警

# ─── 数据 ──────────────────────────────────────────────────
DEFAULT_PERIOD_YEARS = 3       # 拉取历史数据年数
MIN_TRADING_DAYS     = 120     # 最少需要的交易日

# ─── LSTM ──────────────────────────────────────────────────
LSTM_LOOKBACK_SHORT  = 30      # 短期模型回望天数
LSTM_LOOKBACK_LONG   = 60      # 长期模型回望天数
LSTM_EPOCHS          = 60
LSTM_BATCH_SIZE      = 32
LSTM_PATIENCE        = 10      # EarlyStopping

# ─── 预测周期 ───────────────────────────────────────────────
SHORT_TERM_DAYS = 10
LONG_TERM_DAYS  = 30

# ─── 模型融合权重（初始值，运行时按验证RMSE动态调整） ─────────
MODEL_WEIGHTS = {
    "lstm":     0.40,
    "xgboost":  0.35,
    "prophet":  0.25,
}

# ─── 综合评分权重 ───────────────────────────────────────────
SCORE_WEIGHTS = {
    "technical":   0.40,
    "fundamental": 0.30,
    "sentiment":   0.15,
    "prediction":  0.15,
}

# ─── 评级区间 ───────────────────────────────────────────────
RATING_THRESHOLDS = [
    (80, "强烈买入", "Strong Buy",    "#00C853"),
    (65, "买入",     "Buy",           "#69F0AE"),
    (45, "观望",     "Hold",          "#FFD600"),
    (30, "卖出",     "Sell",          "#FF6D00"),
    (0,  "强烈卖出", "Strong Sell",   "#D50000"),
]

# ─── 新闻情感 ───────────────────────────────────────────────
NEWS_LIMIT          = 50       # 每次最多拉取新闻条数
SENTIMENT_POSITIVE  = 0.65    # SnowNLP阈值
SENTIMENT_NEGATIVE  = 0.35
 
# Performance controls
REPORT_USE_PRETRAINED_MODELS = True
REPORT_RETRAIN_IF_MISSING = True
XGB_USE_BAYES_IN_REPORT = False
ONLINE_XGB_USE_BAYES = False

# ── 多核 CPU 利用 ──────────────────────────────────────────
# 设计原则：避免嵌套并行的 oversubscription
#   PARALLEL_STOCK_WORKERS × XGB_N_JOBS ≈ CPU 物理核数
# 1 核机器：1 × 1 = 1，老老实实串行
# 4 核机器：3 × 1 或 2 × 2 = ~4
# 8 核机器：6 × 1 或 4 × 2 = ~8
# 16 核机器:12 × 1 或 8 × 2 = ~16
_CPU = os.cpu_count() or 4
# 多股并行 workers：留 1 核给系统/streamlit，上限 12（避免 ProcessPool 启动开销爆炸）
PARALLEL_STOCK_WORKERS = min(12, max(1, _CPU - 1))
# 单股内部 XGB/sklearn 的 n_jobs：当 stock-level 并行高时，这个必须降下来
#   stocks_workers ≥ CPU/2 → n_jobs=1（避免嵌套）
#   stocks_workers < CPU/2 → n_jobs=2~4
if PARALLEL_STOCK_WORKERS >= _CPU // 2:
    XGB_N_JOBS = 1
else:
    XGB_N_JOBS = max(1, min(4, _CPU // PARALLEL_STOCK_WORKERS))

# 单股内部 ensemble 5 模型 ThreadPool 并行 worker 数
# 注意：模型本身（TF/XGB）也用多线程，此处只是任务调度并发，太大无意义
ENSEMBLE_TRAIN_WORKERS = min(5, max(1, _CPU // 2))

# TensorFlow 线程数（防止与上层 ProcessPool 嵌套时过度订阅）
# 0 = 让 TF 自动决定（=CPU 物理核数），通常单进程跑时最优
# 多进程跑时建议 = max(1, CPU // PARALLEL_STOCK_WORKERS)
TF_INTRAOP_THREADS = max(1, _CPU // max(1, PARALLEL_STOCK_WORKERS))
TF_INTEROP_THREADS = 2


def apply_threading_env():
    """在主进程启动时调用，限制 TF/MKL/OMP 线程数。
    必须在 import tensorflow 之前调用才完全生效（环境变量），
    若 TF 已 import 也会通过 tf.config 强制设置。"""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OMP_NUM_THREADS", str(TF_INTRAOP_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(TF_INTRAOP_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(TF_INTRAOP_THREADS))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(TF_INTRAOP_THREADS))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(TF_INTRAOP_THREADS))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(TF_INTEROP_THREADS))

# Self-learning controls
POSTERIOR_WEIGHT_MAX = 0.25
POSTERIOR_WEIGHT_PER_SAMPLE = 0.05
MODEL_VERSION_KEEP_RECENT = 1
MODEL_VERSION_KEEP_BEST = 1
MODEL_CONTROL_MIN_SAMPLES = 4
MODEL_DISABLE_DEGRADE_STREAK = 3
MODEL_DISABLE_MIN_SAMPLES = 6
MODEL_DOWNWEIGHT_VOLATILE = 0.80
MODEL_DOWNWEIGHT_DEGRADED = 0.45

# Ensemble 权重计算：单模型 RMSE 超过中位数 × 该倍率时强降权
RMSE_OUTLIER_MULTIPLIER = 2.5
# 异常 RMSE 模型保留的权重比例（1.0 = 不降权）
RMSE_OUTLIER_PENALTY = 0.25

# 启动时孤儿模型自动清理（每天最多一次）
AUTO_CLEANUP_ON_STARTUP = True
