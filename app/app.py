"""
A股智能分析系统 — Streamlit Web界面
启动: streamlit run app.py
"""
import sys, os, json, warnings, subprocess
from datetime import datetime
from typing import List, Dict, Any

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# 在线学习框架（可选启用，不影响现有流程）
ONLINE_LEARNING_ENABLED = False  # 默认关闭，用户可手动启用
online_learner = None
scheduler_thread = None

st.set_page_config(
    page_title="A股智能分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",   # 关闭原生侧边栏
)

import config

# 注意：主 streamlit 进程不限制 numpy/MKL 线程数，否则会拖慢 plotly 渲染
# 线程限制只应在 ProcessPool 子进程中应用（见 main.py:_subprocess_init）

# ── 启动时孤儿/0字节模型文件自动清理（每天最多一次）────────────
try:
    from models.online_learner import maybe_startup_cleanup
    _cleanup_result = maybe_startup_cleanup()
    if _cleanup_result and _cleanup_result.get("removed_count", 0) > 0:
        print(f"[startup cleanup] 已清理 {_cleanup_result['removed_count']} 个旧模型，"
              f"释放 {_cleanup_result.get('freed_mb', 0)} MB")
except Exception as _e:
    print(f"[startup cleanup] 跳过: {_e}")

WATCHLIST_FILE = config.WATCHLIST_FILE
BACKGROUND_LEARNING_STATUS_FILE = os.path.join(
    config.PERFORMANCE_LOG_DIR, "_background_learning_status.json"
)


def _write_background_learning_status(payload):
    os.makedirs(os.path.dirname(BACKGROUND_LEARNING_STATUS_FILE), exist_ok=True)
    with open(BACKGROUND_LEARNING_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_background_learning_status():
    if not os.path.exists(BACKGROUND_LEARNING_STATUS_FILE):
        return {}
    try:
        with open(BACKGROUND_LEARNING_STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_pid_running(pid):
    if not pid:
        return False
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (completed.stdout or "").strip()
        return str(pid) in output and "No tasks are running" not in output
    except Exception:
        return False


def _background_learning_runtime_status():
    status = _read_background_learning_status()
    if not status:
        return {}

    started_at = status.get("started_at")
    symbols = status.get("symbols") or []
    total = len(symbols)
    completed_symbols = []
    latest_report = None
    latest_mtime = None
    log_dir = config.PERFORMANCE_LOG_DIR

    if started_at and os.path.isdir(log_dir):
        for symbol in symbols:
            symbol_reports = [
                os.path.join(log_dir, name)
                for name in os.listdir(log_dir)
                if name.startswith(f"{symbol}_update_report_") and name.endswith(".json")
            ]
            for path in symbol_reports:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                except Exception:
                    continue
                report_time = report.get("update_time")
                if not report_time or report_time < started_at:
                    continue
                completed_symbols.append(symbol)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    mtime = 0
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_report = report
                break

    pid = status.get("pid")
    running = _is_pid_running(pid)
    completed_count = len(completed_symbols)

    if running:
        state = "running"
    elif total > 0 and completed_count >= total:
        state = "finished"
    else:
        state = "stopped"

    return {
        **status,
        "running": running,
        "state": state,
        "completed_count": completed_count,
        "total_count": total,
        "latest_symbol": (latest_report or {}).get("symbol"),
        "latest_update_time": (latest_report or {}).get("update_time"),
    }


def _launch_background_learning(symbols):
    current = _background_learning_runtime_status()
    if current.get("state") == "running":
        return {
            "pid": current.get("pid"),
            "command": current.get("command") or [],
            "already_running": True,
        }

    runner = os.path.join(config.BASE_DIR, "run_daily_update.py")
    cmd = [sys.executable, runner, "--symbols", ",".join(symbols)]
    kwargs = {
        "cwd": os.path.dirname(config.BASE_DIR),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    payload = {
        "pid": proc.pid,
        "symbols": symbols,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": cmd,
    }
    _write_background_learning_status(payload)
    return {"pid": proc.pid, "command": cmd, "already_running": False}


def _register_background_learning(symbols):
    from models.online_learner import write_scheduler_helper

    helper = write_scheduler_helper(symbols)
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        helper["register"],
    ]
    completed = subprocess.run(
        cmd,
        cwd=os.path.dirname(config.BASE_DIR),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "注册计划任务失败").strip())
    return {"helper": helper, "stdout": completed.stdout.strip()}


def _unregister_background_learning(task_name="GuangHuanStockDailyUpdate"):
    if os.name != "nt":
        raise RuntimeError("当前平台不支持注销 Windows 计划任务")
    cmd = ["schtasks", "/Delete", "/TN", task_name, "/F"]
    completed = subprocess.run(cmd, capture_output=True, timeout=15)
    if completed.returncode != 0:
        stderr = (completed.stderr or b"").decode("gbk", errors="ignore") or (completed.stdout or b"").decode("gbk", errors="ignore")
        raise RuntimeError((stderr or "注销计划任务失败").strip())
    return {"task_name": task_name}


def _query_scheduled_task(task_name="GuangHuanStockDailyUpdate"):
    if os.name != "nt":
        return {"registered": False, "platform": os.name}
    cmd = ["schtasks", "/Query", "/TN", task_name, "/FO", "LIST", "/V"]
    completed = subprocess.run(cmd, capture_output=True, timeout=15)
    if completed.returncode != 0:
        return {"registered": False}
    raw_stdout = completed.stdout or b""
    for encoding in ("gbk", "utf-8"):
        try:
            stdout_text = raw_stdout.decode(encoding)
            break
        except Exception:
            stdout_text = raw_stdout.decode(encoding, errors="ignore")
            break
    data = {}
    for line in stdout_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return {
        "registered": True,
        "task_name": task_name,
        "status": data.get("Status") or data.get("状态"),
        "next_run": data.get("Next Run Time") or data.get("下次运行时间"),
        "last_run": data.get("Last Run Time") or data.get("上次运行时间"),
        "last_result": data.get("Last Result") or data.get("上次结果"),
    }


def _latest_learning_report_status():
    log_dir = config.PERFORMANCE_LOG_DIR
    if not os.path.isdir(log_dir):
        return {}
    latest_path = None
    latest_mtime = None
    for name in os.listdir(log_dir):
        if not name.endswith(".json") or "_update_report_" not in name:
            continue
        path = os.path.join(log_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    if not latest_path:
        return {}
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return {"path": latest_path}
    return {
        "path": latest_path,
        "symbol": report.get("symbol"),
        "update_time": report.get("update_time"),
        "updated_count": len(report.get("models_updated", []) or []),
        "skipped_count": len(report.get("models_skipped", []) or []),
        "alert_count": len(report.get("alerts", []) or []),
    }


def _mount_risk_expander_style():
    return

# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* 全局 */
.stApp { background:#0D1117; color:#E6EDF3; font-family:'Microsoft YaHei',Arial,sans-serif; }
.block-container { padding-top:1rem !important; padding-bottom:1rem !important; }

/* 所有普通文字强制亮色（不包含 div，避免覆盖彩色卡片） */
p, span, label, li { color:#E6EDF3 !important; }
/* Streamlit 内部文字节点 */
.stMarkdown p, .stText, [data-testid="stText"] { color:#E6EDF3 !important; }

/* 标题 */
h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"],
[data-testid="stHeadingWithActionElements"] { color:#E6EDF3 !important; }

/* Streamlit 组件文字 */
.stCheckbox label, .stSlider label,
.stTextInput label, .stSelectbox label,
[data-testid="stWidgetLabel"] { color:#E6EDF3 !important; font-weight:500 !important; }

/* caption / 说明小字 */
.stCaption, [data-testid="stCaptionContainer"] p { color:#ADBAC7 !important; }

/* 隐藏 Streamlit 自带装饰 */
#MainMenu, footer, header,
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display:none !important; }

/* 左栏（自选股面板）*/
.left-panel {
    background:#161B22; border-right:1px solid #30363D;
    padding:16px 12px;
}

/* 左栏字体缩小 */
[data-testid="stVerticalBlockBorderWrapper"]:first-child {
    font-size:0.85em;
}

/* 激进压缩左栏所有组件间距 */
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stVerticalBlock"] > div,
[data-testid="stVerticalBlockBorderWrapper"]:first-child .stElementContainer,
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="element-container"] {
    margin-top:0 !important; margin-bottom:2px !important;
    padding-top:0 !important; padding-bottom:0 !important;
    gap:0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child .stForm {
    padding:4px !important; margin:0 !important; border:none !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stSlider"] {
    margin:0 0 2px !important; padding:0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stCheckbox"] {
    margin:0 !important; padding:0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child .stButton {
    margin:0 0 2px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child .stButton > button,
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="baseButton-secondary"],
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="baseButton-primary"] {
    padding:4px 8px !important; font-size:0.85em !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) .stButton > button,
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) [data-testid="baseButton-secondary"] {
    background:#FFFFFF !important;
    color:#D1242F !important;
    border:1px solid #D0D7DE !important;
    font-weight:800 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) .stButton > button *,
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) [data-testid="baseButton-secondary"] * {
    color:#D1242F !important;
    fill:#D1242F !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) .stButton > button:hover,
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) [data-testid="baseButton-secondary"]:hover {
    background:#F6F8FA !important;
    color:#A40E26 !important;
    border-color:#8C959F !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) .stButton > button:hover *,
[data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-of-type(2) [data-testid="baseButton-secondary"]:hover * {
    color:#A40E26 !important;
    fill:#A40E26 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child button[title="删除"] {
    background:#FFFFFF !important;
    color:#D1242F !important;
    border:1px solid #D0D7DE !important;
    font-weight:800 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child button[title="删除"] *,
[data-testid="stVerticalBlockBorderWrapper"]:first-child button[aria-label="删除"] *,
[data-testid="stVerticalBlockBorderWrapper"]:first-child button[aria-label*="删除"] * {
    color:#D1242F !important;
    fill:#D1242F !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child button[title="删除"]:hover {
    background:#F6F8FA !important;
    color:#A40E26 !important;
    border-color:#8C959F !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:first-child button[title="删除"]:hover *,
[data-testid="stVerticalBlockBorderWrapper"]:first-child button[aria-label="删除"]:hover *,
[data-testid="stVerticalBlockBorderWrapper"]:first-child button[aria-label*="删除"]:hover * {
    color:#A40E26 !important;
    fill:#A40E26 !important;
}
.panel-title { color:#58A6FF; font-size:1.1em; font-weight:700; margin-bottom:4px; }
.brand-block { margin:0 0 8px; }
.brand-title { color:#FF4D4F; font-size:1.85em; line-height:1.05; font-weight:800; letter-spacing:0; display:flex; align-items:center; gap:6px; }
.brand-icon { font-size:1em; line-height:1; }
.brand-subtitle { color:#ADBAC7; font-size:1.08em; margin-top:4px; font-weight:700; }
.brand-version { color:#58A6FF; font-weight:700; margin-left:6px; }
.section-label { color:#ADBAC7 !important; font-size:0.82em; text-transform:uppercase;
                  letter-spacing:.08em; margin:2px 0 4px; font-weight:600; }

/* 自选股行 */
.wl-row { display:flex; align-items:center; justify-content:space-between;
           padding:5px 0; border-bottom:1px solid #21262D; }
.wl-name { color:#E6EDF3 !important; font-size:0.9em; }
.wl-code { color:#ADBAC7 !important; font-size:0.8em; }
.delete-stock-link {
    display:flex;
    align-items:center;
    justify-content:center;
    width:42px;
    min-height:42px;
    border-radius:8px;
    background:#FFFFFF;
    border:1px solid #D0D7DE;
    color:#D1242F !important;
    text-decoration:none !important;
    font-size:2.5em;
    font-weight:900;
    line-height:1;
}
.delete-stock-link:hover {
    background:#F6F8FA;
    color:#A40E26 !important;
    border-color:#8C959F;
    text-decoration:none !important;
}

/* 评分卡 */
.score-card {
    background:#161B22; border:1px solid #30363D; border-radius:8px;
    padding:14px; text-align:center;
}
.score-card .val { font-size:1.9em; font-weight:700; line-height:1.1; }
.score-card .lbl { color:#ADBAC7 !important; font-size:0.82em; margin-top:4px; }

/* metric 数值和标签 */
[data-testid="stMetricLabel"] { color:#ADBAC7 !important; font-size:0.85em !important; }
[data-testid="stMetricValue"] { color:#E6EDF3 !important; font-weight:700 !important; }
[data-testid="stMetricDelta"] { font-weight:500 !important; }

/* tab 标签 */
.stTabs [data-baseweb="tab"] { color:#ADBAC7 !important; font-size:0.95em; }
.stTabs [aria-selected="true"] { color:#58A6FF !important; font-weight:600 !important; }

/* dataframe 表格 */
[data-testid="stDataFrame"] { color:#E6EDF3 !important; }

/* expander 标题 */
.streamlit-expanderHeader { color:#E6EDF3 !important; font-weight:500 !important; }
[data-testid="stExpander"] details {
    background:#0D1117 !important;
    border:1px solid #30363D !important;
    border-radius:8px !important;
}
[data-testid="stExpander"] details > summary {
    background:#161B22 !important;
    color:#E6EDF3 !important;
    border-radius:8px !important;
}
[data-testid="stExpander"] details > summary * {
    color:#E6EDF3 !important;
    fill:#E6EDF3 !important;
}
[data-testid="stExpander"] details[open] > summary {
    border-bottom:1px solid #30363D !important;
    border-radius:8px 8px 0 0 !important;
}
[data-testid="stCodeBlock"],
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code {
    background:#161B22 !important;
    color:#E6EDF3 !important;
}
[data-testid="stCodeBlock"] {
    border:1px solid #30363D !important;
    border-radius:8px !important;
}
[data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] input {
    color:#0D1117 !important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[role="listbox"],
[role="listbox"] *,
[role="option"],
[role="option"] * {
    background:#F6F8FA !important;
    color:#0D1117 !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background:#E5E7EB !important;
    color:#0D1117 !important;
}

/* 风险管理建议 expander */
[data-testid="stElementContainer"]:has(.risk-expander-anchor) + [data-testid="stElementContainer"] [data-testid="stExpander"] details > summary,
[data-testid="element-container"]:has(.risk-expander-anchor) + [data-testid="element-container"] [data-testid="stExpander"] details > summary {
    background:#D1242F !important;
    border:1px solid #A40E26 !important;
    border-radius:8px !important;
}
[data-testid="stElementContainer"]:has(.risk-expander-anchor) + [data-testid="stElementContainer"] [data-testid="stExpander"] details > summary:hover,
[data-testid="element-container"]:has(.risk-expander-anchor) + [data-testid="element-container"] [data-testid="stExpander"] details > summary:hover {
    background:#0D1117 !important;
    border-color:#30363D !important;
}
[data-testid="stElementContainer"]:has(.risk-expander-anchor) + [data-testid="stElementContainer"] [data-testid="stExpander"] details > summary *,
[data-testid="element-container"]:has(.risk-expander-anchor) + [data-testid="element-container"] [data-testid="stExpander"] details > summary * {
    color:#FFFFFF !important;
    fill:#FFFFFF !important;
}

/* 输入框 */
[data-testid="stTextInput"] input {
    background:#21262D !important; color:#E6EDF3 !important;
    border:1px solid #444C56 !important; border-radius:6px !important;
}
[data-testid="stTextInput"] input::placeholder { color:#768390 !important; }

/* 普通按钮 */
.stButton > button,
[data-testid="baseButton-secondary"] {
    background:#21262D !important; color:#E6EDF3 !important;
    border:1px solid #444C56 !important; border-radius:6px !important;
    font-weight:500 !important;
}
.stButton > button:hover,
[data-testid="baseButton-secondary"]:hover {
    background:#30363D !important; border-color:#58A6FF !important;
    color:#58A6FF !important;
}

/* primary 按钮（分析当前股票）*/
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"] {
    background:#1F6FEB !important; color:#FFFFFF !important;
    border:none !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background:#388BFD !important; color:#FFFFFF !important;
}

/* 表单提交按钮（添加）*/
[data-testid="stFormSubmitButton"] > button {
    background:#21262D !important; color:#E6EDF3 !important;
    border:1px solid #444C56 !important; border-radius:6px !important;
    font-weight:500 !important; width:100% !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background:#30363D !important; border-color:#58A6FF !important;
    color:#58A6FF !important;
}

/* checkbox */
.stCheckbox span { color:#E6EDF3 !important; }

/* slider 数值标签 */
[data-testid="stSlider"] div { color:#E6EDF3 !important; }

/* 分隔线 */
hr { border:none; border-top:1px solid #21262D; margin:3px 0; }

/* ===== 移动端适配 ===== */
@media screen and (max-width: 768px) {

    /* 内边距压缩 */
    .block-container { padding: 0.3rem 0.4rem !important; }

    /* 主布局列 → 垂直堆叠 */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important; gap: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 0 0 100% !important;
        width: 100% !important; min-width: 100% !important; max-width: 100% !important;
    }

    /* 左面板：手机上默认收起，由悬浮按钮控制 */
    #left-panel-mobile { display: none; }
    #left-panel-mobile.open { display: block !important; }
    .left-panel {
        border-right: none !important;
        border-bottom: 1px solid #30363D !important;
        padding: 8px 10px !important;
    }

    /* 触屏按钮加大 */
    .stButton > button {
        min-height: 46px !important; font-size: 1em !important;
    }
    [data-testid="stFormSubmitButton"] > button { min-height: 46px !important; }

    /* 输入框防止 iOS 放大 */
    [data-testid="stTextInput"] input { font-size: 16px !important; min-height: 44px !important; }

    /* 评分卡紧凑 */
    .score-card { padding: 8px 6px !important; }
    .score-card .val { font-size: 1.3em !important; }
    .score-card .lbl { font-size: 0.68em !important; }

    /* Metric */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }

    /* Tab 标签 */
    .stTabs [data-baseweb="tab"] { font-size: 0.78em !important; padding: 8px 4px !important; }

    /* 标题字号 */
    h1 { font-size: 1.3em !important; }
    h2 { font-size: 1.15em !important; }
    h3 { font-size: 1.05em !important; }

    /* 图表不溢出 */
    .js-plotly-plot { max-width: 100% !important; }

    /* 悬浮菜单按钮（仅手机可见）*/
    #mobile-fab {
        display: flex !important;
        position: fixed; bottom: 24px; right: 20px; z-index: 9999;
        width: 52px; height: 52px; border-radius: 50%;
        background: #1F6FEB; color: #fff; font-size: 1.4em;
        align-items: center; justify-content: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.5); cursor: pointer;
        border: none;
    }
}
@media screen and (min-width: 769px) {
    #mobile-fab { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ?? viewport
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_watchlist(wl):
    os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)

def normalize_symbol(s):
    s = s.upper().strip()
    for suffix in (".SS", ".SZ", ".SH", ".BJ", "SH", "SZ"):
        s = s.replace(suffix, "")
    return s.lstrip("0").zfill(6) if s.isdigit() else s

def fmt(val, spec=".1f", suffix=""):
    if val is None: return "—"
    try:    return f"{float(val):{spec}}{suffix}"
    except: return str(val)


def _fast_xgb(df, prediction_days):
    """轻量级 XGBoost 预测（200棵树，快速）"""
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor
    from models.xgboost_model import _build_features

    feat_df = _build_features(df)
    future_ret = (df["close"].shift(-prediction_days) / df["close"] - 1).reindex(feat_df.index)
    mask = future_ret.notna()
    feat_df = feat_df[mask]; target = future_ret[mask]
    if len(feat_df) < 60:
        return None

    feature_cols = feat_df.columns.tolist()
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values); y = target.values
    split = int(len(X) * 0.85)
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=getattr(__import__("config"), "XGB_N_JOBS", 4),
                          early_stopping_rounds=20, eval_metric="rmse")
    model.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])], verbose=False)

    pred_ret_val = model.predict(X[split:])
    close_val = df["close"].reindex(feat_df.index)[mask][split:]
    # 计算收益率 RMSE（而非价格 RMSE），更能反映模型预测能力
    val_rmse_ret = float(np.sqrt(mean_squared_error(y[split:], pred_ret_val)))
    # 同时计算价格 RMSE用于参考
    val_rmse_price = float(np.sqrt(mean_squared_error(
        close_val.values * (1 + y[split:]),
        close_val.values * (1 + pred_ret_val)
    )))

    last_feat = feat_df.iloc[[-1]][feature_cols]
    ret_end = float(model.predict(scaler.transform(last_feat.values))[0])
    last_price = float(df["close"].iloc[-1])
    end_price = last_price * (1 + ret_end)

    predictions = np.linspace(last_price, end_price, prediction_days + 1)[1:]
    # 使用收益率 RMSE 计算噪声和边界（更合理）
    noise = np.random.normal(0, val_rmse_ret * last_price * 0.2, len(predictions))
    predictions = np.clip(predictions + noise, last_price * 0.6, last_price * 1.6)

    dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                           periods=prediction_days, freq="B")
    # 边界也基于收益率 RMSE 计算
    margin = val_rmse_ret * last_price * 1.5
    return {
        "predictions": predictions.tolist(),
        "lower_bound": (predictions - margin).tolist(),
        "upper_bound": (predictions + margin).tolist(),
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "last_price": last_price,
        "pct_change": float((end_price - last_price) / last_price * 100),
        "val_rmse": val_rmse_ret,  # 返回收益率 RMSE
        "val_rmse_price": val_rmse_price,  # 价格 RMSE 供参考
    }


# ══════════════════════════════════════════════════════════════
#  Session State
# ══════════════════════════════════════════════════════════════
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if "selected" not in st.session_state:
    wl0 = st.session_state.watchlist
    st.session_state.selected = wl0[0]["symbol"] if wl0 else None
if "results" not in st.session_state:
    st.session_state.results = {}
if "board_results" not in st.session_state:
    st.session_state.board_results = []


# ══════════════════════════════════════════════════════════════
#  分析核心
# ══════════════════════════════════════════════════════════════
def run_analysis(symbol, years, do_backtest, fast_mode=True):
    r = {"symbol": symbol, "ok": False}
    prog = st.progress(0, "初始化...")
    try:
        prog.progress(5,  "获取历史行情...")
        from data.fetcher import fetch_stock_data, fetch_stock_info, get_stock_name, enrich_with_market_data
        df         = fetch_stock_data(symbol, years=years)
        stock_info = fetch_stock_info(symbol)
        stock_name = get_stock_name(symbol)
        r.update(stock_name=stock_name, stock_info=stock_info,
                 last_price=float(df["close"].iloc[-1]), trading_days=len(df))

        prog.progress(12, "沪深300 + 资金流向...")
        try: df = enrich_with_market_data(df, symbol)
        except: pass

        prog.progress(18, "获取新闻...")
        try:
            from data.news import fetch_news, get_news_texts
            news_list  = fetch_news(symbol)
            news_texts = get_news_texts(news_list)
        except:
            news_list = news_texts = []
        r["news_list"] = news_list

        prog.progress(26, "计算技术指标...")
        from analysis.technical import compute_indicators, generate_signals, compute_support_resistance
        df = compute_indicators(df)
        r.update(df=df, signals=generate_signals(df), sr=compute_support_resistance(df))

        prog.progress(36, "基本面分析...")
        try:
            from analysis.fundamental import fetch_and_analyze
            r["fundamental"] = fetch_and_analyze(symbol, stock_info)
        except:
            r["fundamental"] = {"score": 50, "details": []}

        prog.progress(46, "情感分析...")
        try:
            from analysis.sentiment import analyze_sentiment
            r["sentiment"] = analyze_sentiment(news_texts)
        except:
            r["sentiment"] = {"score":0.5,"label":"中性","positive_count":0,
                               "negative_count":0,"neutral_count":0,"news_count":0}

        if fast_mode:
            prog.progress(55, "训练预测模型（快速模式，约30秒）...")
        else:
            prog.progress(55, "训练预测模型（完整模式，约2分钟）...")

        import config as _cfg
        short_pred = long_pred = None
        try:
            if fast_mode:
                # 快速模式：只用轻量 XGBoost（200棵树，约30秒）
                short_pred = _fast_xgb(df, _cfg.SHORT_TERM_DAYS)
                long_pred  = _fast_xgb(df, _cfg.LONG_TERM_DAYS)
            else:
                # 完整模式：LSTM + XGBoost + Prophet 三模型融合
                from models.ensemble import EnsemblePredictor
                pred = EnsemblePredictor()
                pred.symbol_context = symbol
                pred.posterior_weights = pred._load_posterior_weights(symbol)
                pred.model_control = pred._load_model_control(symbol)
                loaded = pred.load_pretrained(symbol) if getattr(_cfg, "REPORT_USE_PRETRAINED_MODELS", True) else {}
                if not loaded and getattr(_cfg, "REPORT_RETRAIN_IF_MISSING", True):
                    pred.train(df)
                short_pred = pred.predict_short(df)
                long_pred  = pred.predict_long(df)
        except Exception:
            pass
        r.update(short_pred=short_pred, long_pred=long_pred)

        prog.progress(87, "综合评分...")
        from models.scorer import compute_score
        r["score_result"] = compute_score(r["signals"], r["fundamental"], r["sentiment"], short_pred, long_pred)

        prog.progress(89, "价值体检...")
        try:
            from analysis.value_pipeline import build_value_assessment, annotate_signal_conflict
            r["value_assessment"] = build_value_assessment(symbol, stock_info, r["fundamental"])
            annotate_signal_conflict(r["value_assessment"], r["score_result"])
        except Exception as e:
            print(f"[价值体检失败] {e}")

        prog.progress(90, "保存历史...")
        try:
            from data.score_history import save_score, load_history
            from data.prediction_tracker import save_prediction_snapshot
            save_score(symbol, stock_name, r["score_result"], short_pred, long_pred)
            save_prediction_snapshot(
                symbol=symbol,
                stock_name=stock_name,
                last_price=r["last_price"],
                short_pred=short_pred,
                long_pred=long_pred,
                score_result=r["score_result"],
            )
            r["history"] = load_history(symbol)
        except:
            r["history"] = []

        if do_backtest:
            prog.progress(93, "执行回测...")
            try:
                from backtest.engine import run_backtest
                r["backtest"] = run_backtest(df)
            except: r["backtest"] = None
        else:
            r["backtest"] = None

        prog.progress(97, "生成图表...")
        import traceback as _tb
        figs = {}
        chart_errors = []

        def _try_chart(name, fn, *args, **kwargs):
            """单图表生成隔离：一个图失败不影响其他"""
            try:
                fig = fn(*args, **kwargs)
                if fig is not None:
                    figs[name] = fig
            except Exception as _e:
                err = f"{name}: {_e}"
                chart_errors.append(err)
                print(f"[图表生成失败] {err}")
                _tb.print_exc()

        try:
            from visualization.charts import (
                create_candlestick_chart, create_prediction_chart,
                create_score_gauge, create_fundamental_radar,
                create_signal_bar, create_score_history_chart,
            )
            sr = r["score_result"]
            _try_chart("candle",  create_candlestick_chart, df, stock_name, symbol)
            _try_chart("predict", create_prediction_chart,  df, short_pred, long_pred, stock_name)
            _try_chart("gauge",   create_score_gauge,       sr["total_score"], sr["rating"], sr["color"])
            _try_chart("radar",   create_fundamental_radar, r["fundamental"], stock_name)
            if r["signals"].get("signals"):
                _try_chart("signals", create_signal_bar, r["signals"]["signals"])
            if len(r.get("history", [])) > 1:
                _try_chart("history", create_score_history_chart, r["history"], stock_name)
        except Exception as _import_exc:
            print(f"[图表模块导入失败] {_import_exc}")
            _tb.print_exc()
            chart_errors.append(f"import: {_import_exc}")

        r["figs"] = figs
        if chart_errors:
            r["_chart_error"] = "; ".join(chart_errors[:3])

        r["ok"] = True
        r["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        r["error"] = str(e)
    prog.progress(100, "完成")
    prog.empty()
    return r


def run_quick(symbol):
    try:
        from data.fetcher import fetch_stock_data, fetch_stock_info, get_stock_name
        from analysis.technical import compute_indicators, generate_signals
        from analysis.fundamental import fetch_and_analyze
        from analysis.sentiment import analyze_sentiment
        from data.news import fetch_news, get_news_texts
        from models.scorer import compute_score
        from models.ensemble import EnsemblePredictor

        df   = fetch_stock_data(symbol, years=2)
        info = fetch_stock_info(symbol)
        name = get_stock_name(symbol)
        df   = compute_indicators(df)
        sig  = generate_signals(df)
        fund = fetch_and_analyze(symbol, info)
        sent = analyze_sentiment(get_news_texts(fetch_news(symbol, limit=10)))
        short_pred = long_pred = None
        try:
            p = EnsemblePredictor()
            p.symbol_context = sym
            p.posterior_weights = p._load_posterior_weights(sym)
            p.model_control = p._load_model_control(sym)
            p.train(df)
            short_pred = p.predict_short(df)
            long_pred  = p.predict_long(df)
        except: pass
        res = compute_score(sig, fund, sent, short_pred, long_pred)
        return {"symbol": symbol, "name": name, "score_result": res,
                "short_pred": short_pred, "long_pred": long_pred,
                "last_price": float(df["close"].iloc[-1])}
    except Exception as e:
        return {"symbol": symbol, "name": "—", "error": str(e)}


# ══════════════════════════════════════════════════════════════
#  布局：左栏 + 右栏（用 columns 代替 sidebar）
# ══════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 6], gap="small")

# ─────────────────────────────────────────────────────────────
#  左栏：自选股 + 设置
# ─────────────────────────────────────────────────────────────
with col_left:
    wl = st.session_state.watchlist
    sidebar_notice = st.empty()

    app_version = getattr(config, "APP_VERSION", "v1.0.0")
    st.markdown(
        f'<div class="brand-block"><div class="brand-title"><span class="brand-icon">📫</span>光环智能</div><div class="brand-subtitle">股票预测系统 <span class="brand-version">{app_version}</span></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">🚀 操作</div>', unsafe_allow_html=True)
    if st.button('📈 分析当前股票', width='stretch', type='primary', key='top_analyze_button'):
        selected_symbol = st.session_state.get('selected')
        valid_symbols = {item['symbol'] for item in wl}
        if selected_symbol and selected_symbol in valid_symbols:
            st.session_state._do_analyze = True
            st.session_state._do_analyze_symbol = selected_symbol
        else:
            sidebar_notice.warning('请先选择股票')

    if st.button('📊 多股对比看板', width='stretch', key='top_board_button'):
        if wl:
            st.session_state._do_board = True
        else:
            sidebar_notice.warning('请先添加股票')

    st.markdown('<div class="section-label">📌 自选股</div>', unsafe_allow_html=True)
    if not wl:
        st.caption('暂无自选股，请先添加')

    for i, item in enumerate(wl):
        sym_i = item['symbol']
        name_i = item.get('name', sym_i)
        is_sel = sym_i == st.session_state.selected
        c1, c2 = st.columns([5, 1])
        with c1:
            marker = chr(0x25B6) if is_sel else chr(0x2007)
            label = f"{marker} {name_i}  `{sym_i}`"
            if st.button(label, key=f'sel_{i}', width='stretch', type='primary' if is_sel else 'secondary'):
                st.session_state.selected = sym_i
                st.session_state.pop('_do_analyze', None)
                st.session_state.pop('_do_analyze_symbol', None)
                st.rerun()
        with c2:
            if st.button('×', key=f'del_{sym_i}', width='stretch'):
                st.session_state.watchlist = [w for w in wl if w['symbol'] != sym_i]
                save_watchlist(st.session_state.watchlist)
                if st.session_state.selected == sym_i:
                    st.session_state.selected = st.session_state.watchlist[0]['symbol'] if st.session_state.watchlist else None
                st.rerun()

    st.markdown('<div class="section-label">➕ 添加股票</div>', unsafe_allow_html=True)
    st.caption('支持批量输入，用空格 / 逗号 / 换行分隔')
    with st.form('add_form', clear_on_submit=True):
        new_code = st.text_area(
            '股票代码',
            placeholder='如：600519 000858 601318\n或逗号：600519,000858',
            height=80,
            label_visibility='collapsed',
            help='沪市一般以 6 开头；深市/创业板一般以 0/3 开头；科创板一般以 688 开头。',
        )
        if st.form_submit_button('添加', width='stretch'):
            import re
            raw_codes = re.split(r'[\s,，、；\n]+', new_code.strip())
            raw_codes = [c.strip() for c in raw_codes if c.strip()]
            if not raw_codes:
                sidebar_notice.error('请输入股票代码')
            else:
                from data.fetcher import fetch_stock_data, get_stock_name
                added, skipped, failed = [], [], []
                prog = st.progress(0, text=f'共 {len(raw_codes)} 只，验证中...')
                for i, raw in enumerate(raw_codes):
                    sym = normalize_symbol(raw)
                    prog.progress((i + 1) / len(raw_codes), text=f'验证 {sym}...')
                    if any(w['symbol'] == sym for w in wl):
                        skipped.append(sym)
                        continue
                    try:
                        df_test = fetch_stock_data(sym, years=1)
                        if df_test is None or df_test.empty:
                            raise ValueError()
                        name = get_stock_name(sym)
                        wl.append({'symbol': sym, 'name': name})
                        added.append(f'{name}({sym})')
                    except Exception:
                        failed.append(sym)
                prog.empty()
                if added:
                    save_watchlist(wl)
                    if not st.session_state.selected:
                        st.session_state.selected = wl[0]['symbol']
                    sidebar_notice.success(f'已添加 {len(added)} 只：{"、".join(added)}')
                    st.rerun()
                elif skipped:
                    sidebar_notice.warning(f'已存在：{"、".join(skipped)}')
                elif failed:
                    sidebar_notice.error(f'无效代码：{"、".join(failed)}')

    st.markdown('<div class="section-label">⚙️ 分析设置</div>', unsafe_allow_html=True)
    years = st.slider('历史年数', 1, 5, 3, label_visibility='visible')
    do_bt = st.checkbox('执行回测', value=True)
    fast_mode = st.checkbox(
        '快速模式',
        value=False,
        help='开启后优先使用较快模型；关闭后使用完整模型组合',
    )

    st.markdown('<div class="section-label">🤖 在线学习</div>', unsafe_allow_html=True)
    online_enabled = st.checkbox(
        '实现在线学习',
        value=False,
        help='启用后可手动触发后台自学习，也可通过计划任务执行每日自学习',
        key='online_learning_toggle',
    )

    if online_enabled and not ONLINE_LEARNING_ENABLED:
        try:
            from models.online_learner import setup_scheduler

            ONLINE_LEARNING_ENABLED = True
            watchlist = st.session_state.get('watchlist', [])
            if watchlist:
                symbols = [item['symbol'] for item in watchlist]
                scheduler_info = setup_scheduler(symbols)
                if isinstance(scheduler_info, dict) and scheduler_info.get('mode') == 'external_only':
                    sidebar_notice.success('在线学习已授权，已生成自主调度脚本，可通过系统计划任务执行每日自学习')
                else:
                    sidebar_notice.success('在线学习已启用，将在每个交易日自动更新模型')
            else:
                sidebar_notice.warning('请先添加自选股，然后重新启用在线学习')
                ONLINE_LEARNING_ENABLED = False
        except Exception as e:
            sidebar_notice.error(f'启用在线学习失败: {e}')
            ONLINE_LEARNING_ENABLED = False
    elif not online_enabled and ONLINE_LEARNING_ENABLED:
        ONLINE_LEARNING_ENABLED = False
        sidebar_notice.info('在线学习已禁用')

    if st.button('🚀 立即后台执行在线学习', disabled=not online_enabled, width='stretch', key='sidebar_launch_online_learning'):
        try:
            watchlist = st.session_state.get('watchlist', [])
            if watchlist:
                symbols = [item['symbol'] for item in watchlist]
                launch_info = _launch_background_learning(symbols)
                if launch_info.get('already_running'):
                    sidebar_notice.warning(f'后台在线学习已在运行，进程 PID: {launch_info["pid"]}')
                else:
                    sidebar_notice.success(f'后台在线学习已启动，进程 PID: {launch_info["pid"]}')
            else:
                sidebar_notice.warning('请先添加自选股')
        except Exception as e:
            sidebar_notice.error(f'启动在线学习失败: {e}')

    task_status_brief = _query_scheduled_task()
    latest_learning_brief = _latest_learning_report_status()
    bg_runtime = _background_learning_runtime_status()
    st.caption(f'计划任务：{"已注册" if task_status_brief.get("registered") else "未注册"}')
    st.caption(f'上次后台学习：{latest_learning_brief.get("update_time") or "暂无"}')
    if bg_runtime:
        if bg_runtime.get('state') == 'running':
            st.info(
                f'后台学习进行中：{bg_runtime.get("completed_count", 0)}/{bg_runtime.get("total_count", 0)}'
                + (f'，最近完成 {bg_runtime.get("latest_symbol")}' if bg_runtime.get('latest_symbol') else '')
            )
        elif bg_runtime.get('state') == 'finished':
            st.success(f'最近一次后台学习已完成：{bg_runtime.get("completed_count", 0)}/{bg_runtime.get("total_count", 0)}')
        elif bg_runtime.get('state') == 'stopped':
            st.warning(f'后台学习进程已结束：{bg_runtime.get("completed_count", 0)}/{bg_runtime.get("total_count", 0)}')

with col_right:
    tab_analyze, tab_board, tab_history, tab_online = st.tabs(["📊 分析报告", "🏆 多股看板", "📅 历史评分", "🤖 在线学习"])

    # ══ Tab 1: 分析报告 ══════════════════════════════════════
    with tab_analyze:
        sym = st.session_state.selected

        if not sym:
            st.info("👈 在左侧添加并选择股票，然后点击「分析当前股票」")

        else:
            # 执行分析
            if st.session_state.pop("_do_analyze", False):
                analyze_symbol = st.session_state.pop("_do_analyze_symbol", None)
                if analyze_symbol == sym:
                    config.DEFAULT_PERIOD_YEARS = years
                    eta = "\u7ea630\u79d2" if fast_mode else "\u7ea61-2\u5206\u949f"
                    with st.spinner(f"\u6b63\u5728\u5206\u6790 {sym}\uff0c{eta}..."):
                        st.session_state.results[sym] = run_analysis(sym, years, do_bt, fast_mode=fast_mode)
                    st.rerun()
                else:
                    st.warning("\u8bf7\u5148\u786e\u8ba4\u5df2\u9009\u4e2d\u7684\u80a1\u7968\uff0c\u518d\u751f\u6210\u62a5\u544a")

            r = st.session_state.results.get(sym)

            if not r:
                st.info(f"已选中 **{sym}**，点击左侧「🔍 分析当前股票」开始")

            elif not r.get("ok"):
                st.error(f"分析失败：{r.get('error','未知错误')}")
                st.info("💡 请检查股票代码是否正确，或从左侧列表删除该股票后重新添加")
                if st.button("🗑️ 从自选股删除此股票"):
                    wl = st.session_state.watchlist
                    st.session_state.watchlist = [w for w in wl if w["symbol"] != sym]
                    save_watchlist(st.session_state.watchlist)
                    st.session_state.selected = st.session_state.watchlist[0]["symbol"] if st.session_state.watchlist else None
                    del st.session_state.results[sym]
                    st.rerun()

            else:
                sr   = r["score_result"]
                figs = r.get("figs", {})

                # ── 颜色工具 ──────────────────────────────────
                def score_color(v):
                    if v >= 80: return "#22c55e"
                    if v >= 65: return "#4CAF50"
                    if v >= 45: return "#FFA726"
                    if v >= 30: return "#EF5350"
                    return "#f44336"

                def pct_color(v):
                    # A股：涨红跌绿
                    return "#EF5350" if v > 0 else ("#22c55e" if v < 0 else "#ADBAC7")

                def cmet(label, value, color, sub="", size="1.35em", label_size="0.85em"):
                    sub_html = f'<div style="color:#ADBAC7;font-size:0.82em;margin-top:4px">{sub}</div>' if sub else ""
                    return (f'<div class="score-card" style="text-align:left;padding:12px 16px">'
                            f'<div style="color:#ADBAC7;font-size:{label_size};margin-bottom:6px">{label}</div>'
                            f'<div style="color:{color};font-size:{size};font-weight:700;line-height:1.1">{value}</div>'
                            f'{sub_html}</div>')

                def confidence_card(value, color, sub=""):
                    sub_html = f'<div style="color:#ADBAC7;font-size:0.9em;margin-top:6px;font-weight:700">{sub}</div>' if sub else ""
                    return (f'<div class="score-card" style="text-align:center;padding:14px 16px">'
                            f'<div style="color:#ADBAC7;font-size:0.9em;margin-bottom:6px">报告可信度</div>'
                            f'<div style="color:{color};font-size:2.7em;font-weight:800;line-height:1">{value}</div>'
                            f'{sub_html}</div>')

                def _report_nrmse_score(nrmse_pct):
                    if nrmse_pct <= 2:
                        return 95
                    if nrmse_pct <= 4:
                        return 80
                    if nrmse_pct <= 6:
                        return 65
                    if nrmse_pct <= 8:
                        return 50
                    if nrmse_pct <= 12:
                        return 35
                    return 20

                def _report_confidence_grade(score):
                    if score >= 85:
                        return "高可信", "#22c55e"
                    if score >= 70:
                        return "较可信", "#4CAF50"
                    if score >= 55:
                        return "可参考", "#d29922"
                    if score >= 40:
                        return "谨慎参考", "#f97316"
                    return "低可信", "#f85149"

                def _collect_saved_rmse(symbol_):
                    version_file = os.path.join(config.ONLINE_MODEL_DIR, "model_versions.json")
                    if not os.path.exists(version_file):
                        return {}
                    try:
                        with open(version_file, "r", encoding="utf-8") as f:
                            all_versions = json.load(f)
                    except Exception:
                        return {}

                    saved = {}
                    for model_key in ("lstm_short", "lstm_long", "xgb_short", "xgb_long", "prophet"):
                        versions = all_versions.get(f"{symbol_}_{model_key}", [])
                        valid = [
                            item.get("rmse")
                            for item in versions
                            if isinstance(item.get("rmse"), (int, float)) and item.get("rmse") < float("inf")
                        ]
                        if valid:
                            saved[model_key] = min(valid)
                    return saved

                def _collect_report_rmse(sp_, lp_, symbol_):
                    items = {}
                    if sp_ and sp_.get("model_rmse"):
                        items.update({k: v for k, v in sp_["model_rmse"].items() if v < float("inf")})
                    if lp_ and lp_.get("model_rmse"):
                        items.update({k: v for k, v in lp_["model_rmse"].items() if v < float("inf")})
                    if sp_ and sp_.get("val_rmse_price"):
                        items["xgb_short"] = sp_.get("val_rmse_price")
                    if lp_ and lp_.get("val_rmse_price"):
                        items["xgb_long"] = lp_.get("val_rmse_price")
                    for key, val in _collect_saved_rmse(symbol_).items():
                        items.setdefault(key, val)
                    return items

                def _calc_report_confidence_common(rmse_map, price):
                    rmse_order_ = [
                        ("lstm_short", "LSTM-S", 0.20),
                        ("lstm_long", "LSTM-L", 0.20),
                        ("xgb_short", "XGB-S", 0.25),
                        ("xgb_long", "XGB-L", 0.25),
                        ("prophet", "Prophet", 0.10),
                    ]
                    if not price or price <= 0:
                        return None
                    model_error = 0.0
                    valid = []
                    abnormal = []
                    detail = {}
                    for key, label, weight in rmse_order_:
                        rmse_val = rmse_map.get(key)
                        if rmse_val is None:
                            continue
                        nrmse_pct = float(rmse_val) / float(price) * 100
                        item_score = _report_nrmse_score(nrmse_pct)
                        model_error += item_score * weight
                        valid.append(float(rmse_val))
                        detail[key] = {"label": label, "nrmse_pct": nrmse_pct, "score": item_score}
                        if nrmse_pct > 12:
                            abnormal.append(label)
                    if not valid:
                        return None
                    if len(valid) >= 2:
                        mean_rmse = sum(valid) / len(valid)
                        std_rmse = (sum((v - mean_rmse) ** 2 for v in valid) / len(valid)) ** 0.5
                        cv = std_rmse / mean_rmse if mean_rmse else 0
                        consistency = 100 - min(60, cv * 100)
                    else:
                        consistency = 0
                    completeness = len(valid) / 5 * 100
                    posterior = {
                        "count": 0,
                        "avg_relative_error_pct": None,
                        "direction_accuracy_pct": None,
                        "interval_hit_rate_pct": None,
                        "score": None,
                    }
                    try:
                        from data.prediction_tracker import summarize_posterior_metrics, summarize_posterior_health
                    except Exception:
                        try:
                            from app.data.prediction_tracker import summarize_posterior_metrics, summarize_posterior_health
                        except Exception:
                            summarize_posterior_metrics = None
                            summarize_posterior_health = None
                    if summarize_posterior_metrics is not None:
                        posterior = summarize_posterior_metrics(sym)
                        if posterior.get("count", 0) > 0:
                            posterior_error = posterior.get("avg_relative_error_pct")
                            posterior_dir = posterior.get("direction_accuracy_pct")
                            posterior_hit = posterior.get("interval_hit_rate_pct")
                            posterior_score = 0.0
                            parts = 0
                            if posterior_error is not None:
                                posterior_score += _report_nrmse_score(posterior_error)
                                parts += 1
                            if posterior_dir is not None:
                                posterior_score += posterior_dir
                                parts += 1
                            if posterior_hit is not None:
                                posterior_score += posterior_hit
                                parts += 1
                            posterior["score"] = posterior_score / parts if parts else None
                    health = {
                        "status": "样本不足",
                        "summary": "暂无足够后验样本",
                        "color": "#ADBAC7",
                    }
                    if summarize_posterior_health is not None:
                        health = summarize_posterior_health(sym)
                    health_bonus = 0.0
                    if health.get("status") == "稳定提升":
                        health_bonus = 6.0
                    elif health.get("status") == "基本稳定":
                        health_bonus = 2.0
                    elif health.get("status") == "波动加大":
                        health_bonus = -6.0
                    elif health.get("status") == "明显退化":
                        health_bonus = -12.0

                    score = model_error * 0.70 + consistency * 0.20 + completeness * 0.10
                    if posterior.get("score") is not None:
                        posterior_weight = min(
                            getattr(config, "POSTERIOR_WEIGHT_MAX", 0.25),
                            getattr(config, "POSTERIOR_WEIGHT_PER_SAMPLE", 0.05) * posterior.get("count", 0),
                        )
                        score = score * (1 - posterior_weight) + posterior["score"] * posterior_weight
                    score = max(0.0, min(100.0, score + health_bonus))
                    grade, color = _report_confidence_grade(score)
                    return {
                        "score": score,
                        "grade": grade,
                        "color": color,
                        "model_error": model_error,
                        "consistency": consistency,
                        "completeness": completeness,
                        "abnormal": abnormal,
                        "detail": detail,
                        "posterior": posterior,
                        "health": health,
                    }

                # ── 标题 ──────────────────────────────────────
                sp = r.get("short_pred")
                lp = r.get("long_pred")
                report_rmse_items = _collect_report_rmse(sp, lp, sym)
                report_confidence = _calc_report_confidence_common(report_rmse_items, r.get("last_price"))
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"## {r['stock_name']}（{sym}）")
                    st.caption(f"分析时间：{r.get('ts','')}  ·  {r['trading_days']} 个交易日")
                    conf_col, _ = st.columns([1, 2])
                    with conf_col:
                        if report_confidence:
                            st.markdown(confidence_card(f"{report_confidence['score']:.1f}",
                                                        report_confidence["color"],
                                                        report_confidence["grade"]),
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(confidence_card("--", "#ADBAC7"),
                                        unsafe_allow_html=True)
                with c2:
                    st.markdown(cmet("当前价格", f"¥{r['last_price']:.2f}",
                                     "#E6EDF3", size="2.6em", label_size="1em"),
                                unsafe_allow_html=True)
                with c3:
                    if sp:
                        pct  = sp["pct_change"]
                        arrow = "▲" if pct >= 0 else "▼"
                        st.markdown(cmet("10天预测", f"¥{sp['predictions'][-1]:.2f}",
                                         pct_color(pct),
                                         f"{arrow} {abs(pct):.2f}%",
                                         size="2.6em", label_size="1em"),
                                    unsafe_allow_html=True)
                st.divider()

                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                cards = [
                    (sc1, sr["total_score"],       f"综合评分 · {sr['rating']}"),
                    (sc2, sr["technical_score"],   "技术面 (40%)"),
                    (sc3, sr["fundamental_score"], "基本面 (30%)"),
                    (sc4, sr["sentiment_score"],   "情感面 (15%)"),
                    (sc5, sr["prediction_score"],  "预测面 (15%)"),
                ]
                for col_, val_, lbl_ in cards:
                    clr_ = score_color(val_)
                    with col_:
                        st.markdown(

                            f'<div class="score-card">'
                            f'<div class="val" style="color:{clr_}">{val_:.1f}</div>'
                            f'<div class="lbl">{lbl_}</div></div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("")

                def _render_rmse_panel():
                    sp_ = r.get("short_pred")
                    lp_ = r.get("long_pred")
                    rmse_items = {}

                    def _load_saved_rmse(symbol_):
                        version_file = os.path.join(config.ONLINE_MODEL_DIR, "model_versions.json")
                        if not os.path.exists(version_file):
                            return {}
                        try:
                            with open(version_file, "r", encoding="utf-8") as f:
                                all_versions = json.load(f)
                        except Exception:
                            return {}

                        saved = {}
                        for model_key in ("lstm_short", "lstm_long", "xgb_short", "xgb_long", "prophet"):
                            versions = all_versions.get(f"{symbol_}_{model_key}", [])
                            valid = [
                                item.get("rmse")
                                for item in versions
                                if isinstance(item.get("rmse"), (int, float)) and item.get("rmse") < float("inf")
                            ]
                            if valid:
                                saved[model_key] = min(valid)
                        return saved

                    def _nrmse_score(nrmse_pct):
                        if nrmse_pct <= 2:
                            return 95
                        if nrmse_pct <= 4:
                            return 80
                        if nrmse_pct <= 6:
                            return 65
                        if nrmse_pct <= 8:
                            return 50
                        if nrmse_pct <= 12:
                            return 35
                        return 20

                    def _confidence_grade(score):
                        if score >= 85:
                            return "高可信", "#22c55e"
                        if score >= 70:
                            return "较可信", "#4CAF50"
                        if score >= 55:
                            return "可参考", "#d29922"
                        if score >= 40:
                            return "谨慎参考", "#f97316"
                        return "低可信", "#f85149"

                    def _calc_report_confidence(rmse_map, price):
                        return _calc_report_confidence_common(rmse_map, price)

                    if sp_ and sp_.get("model_rmse"):
                        rmse_items.update({k: v for k, v in sp_["model_rmse"].items() if v < float("inf")})
                    if lp_ and lp_.get("model_rmse"):
                        rmse_items.update({k: v for k, v in lp_["model_rmse"].items() if v < float("inf")})

                    if sp_ and sp_.get("val_rmse_price"):
                        rmse_items["xgb_short"] = sp_.get("val_rmse_price")
                    if lp_ and lp_.get("val_rmse_price"):
                        rmse_items["xgb_long"] = lp_.get("val_rmse_price")

                    saved_rmse = _load_saved_rmse(sym)
                    for key, val in saved_rmse.items():
                        rmse_items.setdefault(key, val)

                    if not rmse_items:
                        return

                    confidence = _calc_report_confidence(rmse_items, r.get("last_price"))

                    with st.expander("📊 报告可信度评估 / 各模型验证 (RMSE)"):
                        if confidence:
                            abnormal_text = "、".join(confidence["abnormal"]) if confidence["abnormal"] else "无"
                            cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                            cc1.markdown(cmet("报告可信度", f"{confidence['score']:.1f}", confidence["color"], confidence["grade"], size="2.0em"), unsafe_allow_html=True)
                            cc2.markdown(cmet("模型误差", f"{confidence['model_error']:.1f}", score_color(confidence["model_error"])), unsafe_allow_html=True)
                            cc3.markdown(cmet("一致性", f"{confidence['consistency']:.1f}", score_color(confidence["consistency"])), unsafe_allow_html=True)
                            cc4.markdown(cmet("完整度", f"{confidence['completeness']:.0f}%", score_color(confidence["completeness"]), f"异常：{abnormal_text}"), unsafe_allow_html=True)
                            posterior = confidence.get("posterior", {}) or {}
                            health = confidence.get("health", {}) or {}
                            posterior_score = posterior.get("score")
                            posterior_sub = health.get("status") or (f"{posterior.get('count', 0)}次后验" if posterior.get("count", 0) else "暂无后验")
                            cc5.markdown(
                                cmet(
                                    "后验表现",
                                    f"{posterior_score:.1f}" if posterior_score is not None else "--",
                                    score_color(posterior_score) if posterior_score is not None else "#ADBAC7",
                                    posterior_sub,
                                ),
                                unsafe_allow_html=True,
                            )
                            st.markdown("")
                        rmse_order = [
                            ("lstm_short", "LSTM-S"),
                            ("lstm_long", "LSTM-L"),
                            ("xgb_short", "XGB-S"),
                            ("xgb_long", "XGB-L"),
                            ("prophet", "Prophet"),
                        ]
                        cols = st.columns(5)
                        for i, (key, display_name) in enumerate(rmse_order):
                            rmse_val = rmse_items.get(key)
                            with cols[i]:
                                if rmse_val is None:
                                    st.markdown(cmet(display_name, "--", "#ADBAC7"), unsafe_allow_html=True)
                                else:
                                    nrmse_pct = (confidence or {}).get("detail", {}).get(key, {}).get("nrmse_pct")
                                    is_abnormal = nrmse_pct is not None and nrmse_pct > 12
                                    rmse_color = "#f85149" if is_abnormal else "#3fb950" if rmse_val < 2 else "#d29922" if rmse_val < 4 else "#f85149"
                                    sub = f"相对误差 {nrmse_pct:.2f}%" if nrmse_pct is not None else ""
                                    if is_abnormal:
                                        sub = f"{sub} · 异常"
                                    st.markdown(cmet(display_name, f"{rmse_val:.2f}", rmse_color, sub), unsafe_allow_html=True)
                        st.caption("RMSE 越低越好：<2 优秀 🟢 | 2-4 良好 🟡 | >4 偏差 🔴")

                _render_rmse_panel()

                def _render_prediction_review_panel():
                    try:
                        from data.prediction_tracker import (
                            build_prediction_review_rows,
                            evaluate_matured_predictions,
                            load_prediction_records,
                        )
                    except Exception:
                        try:
                            from app.data.prediction_tracker import (
                                build_prediction_review_rows,
                                evaluate_matured_predictions,
                                load_prediction_records,
                            )
                        except Exception as e:
                            st.warning(f"历史预测回顾模块加载失败：{e}")
                            return

                    try:
                        if r.get("df") is not None:
                            evaluate_matured_predictions(sym, r["df"])
                    except Exception:
                        pass

                    rows = build_prediction_review_rows(sym, latest_price=r.get("last_price"), limit=30)

                    with st.expander("📚 历史预测回顾（与最新价格对比）", expanded=False):
                        if not rows:
                            st.info("暂无历史预测记录。运行几次「分析当前股票」后，这里会自动积累并回顾预测效果。")
                            return

                        import pandas as pd

                        df_review = pd.DataFrame(rows)
                        valid_errors = [
                            float(x)
                            for x in df_review["误差%"].dropna().tolist()
                            if isinstance(x, (int, float))
                        ]
                        direction_items = df_review[df_review["方向判断"].isin(["正确", "错误"])]
                        direction_accuracy = None
                        if not direction_items.empty:
                            direction_accuracy = (
                                (direction_items["方向判断"] == "正确").sum()
                                / len(direction_items)
                                * 100
                            )

                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("历史预测", f"{len(df_review)} 条")
                        mc2.metric("已正式验证", f"{int((df_review['状态'] == '已验证').sum())} 条")
                        mc3.metric(
                            "平均误差",
                            f"{sum(valid_errors) / len(valid_errors):.2f}%" if valid_errors else "—",
                        )
                        mc4.metric(
                            "方向正确率",
                            f"{direction_accuracy:.0f}%" if direction_accuracy is not None else "—",
                        )

                        display_cols = [
                            "分析时间",
                            "周期",
                            "状态",
                            "当时价格",
                            "预测价",
                            "目标日期",
                            "对比日期",
                            "实际/最新价",
                            "误差%",
                            "预测方向",
                            "方向判断",
                            "综合评分",
                            "评级",
                        ]
                        st.dataframe(
                            df_review[display_cols],
                            hide_index=True,
                            width="stretch",
                            key=f"prediction_review_table_{sym}",
                        )
                        st.caption("说明：未到目标日期的记录使用最新价格做阶段对比；到达目标日期后会锁定为正式验证结果。")

                        options = list(range(len(rows)))

                        def _row_label(i):
                            row = rows[i]
                            return f"{row['分析时间']} · {row['周期']} · {row['状态']} · 预测{row['预测价']}"

                        selected_idx = st.selectbox(
                            "查看单次预测明细",
                            options=options,
                            format_func=_row_label,
                            key=f"prediction_review_detail_select_{sym}",
                        )
                        selected = rows[selected_idx]
                        records = load_prediction_records(sym)
                        selected_record = next(
                            (
                                item
                                for item in records
                                if item.get("snapshot_id") == selected.get("_snapshot_id")
                            ),
                            None,
                        )
                        if not selected_record:
                            return

                        pred_block = (
                            (selected_record.get("predictions") or {})
                            .get(selected.get("_horizon"))
                            or {}
                        )
                        score_snapshot = selected_record.get("score_result") or {}
                        d1, d2, d3 = st.columns(3)
                        d1.markdown(
                            cmet(
                                "当时价格",
                                f"¥{selected.get('当时价格'):.2f}" if selected.get("当时价格") is not None else "—",
                                "#E6EDF3",
                                selected_record.get("generated_at", ""),
                            ),
                            unsafe_allow_html=True,
                        )
                        d2.markdown(
                            cmet(
                                "预测目标",
                                f"¥{selected.get('预测价'):.2f}" if selected.get("预测价") is not None else "—",
                                pct_color(pred_block.get("pct_change", 0)),
                                f"{pred_block.get('pct_change', 0):+.2f}% · {selected.get('目标日期')}",
                            ),
                            unsafe_allow_html=True,
                        )
                        d3.markdown(
                            cmet(
                                "实际/最新",
                                f"¥{selected.get('实际/最新价'):.2f}" if selected.get("实际/最新价") is not None else "—",
                                "#22c55e" if selected.get("方向判断") == "正确" else "#f85149" if selected.get("方向判断") == "错误" else "#ADBAC7",
                                f"误差 {selected.get('误差%')}% · {selected.get('方向判断')}",
                            ),
                            unsafe_allow_html=True,
                        )

                        if score_snapshot:
                            st.markdown("**当时评分快照**")
                            sca, scb, scc, scd, sce = st.columns(5)
                            sca.metric("综合", score_snapshot.get("total_score", "—"))
                            scb.metric("技术", score_snapshot.get("technical_score", "—"))
                            scc.metric("基本面", score_snapshot.get("fundamental_score", "—"))
                            scd.metric("情感", score_snapshot.get("sentiment_score", "—"))
                            sce.metric("预测", score_snapshot.get("prediction_score", "—"))

                        components = pred_block.get("component_predictions") or {}
                        if components:
                            comp_rows = []
                            for model_key, comp in components.items():
                                comp_rows.append({
                                    "模型": model_key,
                                    "预测价": round(float(comp.get("target_price", 0) or 0), 2),
                                    "下界": round(float(comp.get("lower_bound", 0) or 0), 2) if comp.get("lower_bound") is not None else None,
                                    "上界": round(float(comp.get("upper_bound", 0) or 0), 2) if comp.get("upper_bound") is not None else None,
                                })
                            st.markdown("**模型分项预测**")
                            st.dataframe(
                                pd.DataFrame(comp_rows),
                                hide_index=True,
                                width="stretch",
                                key=f"prediction_review_components_{sym}_{selected_idx}",
                            )

                _render_prediction_review_panel()

                # ── 回测验证：用历史数据检验模型真实方向准确率 ──
                st.markdown("---")
                st.subheader("📐 模型回测 · 方向准确率验证")
                try:
                    from analysis.backtest import (
                        get_latest_backtest, format_summary_line, run_backtest,
                    )
                except Exception:
                    get_latest_backtest = run_backtest = None

                if not get_latest_backtest:
                    st.caption("回测模块未就绪")
                else:
                    _bt = get_latest_backtest(sym)
                    _bh = (_bt or {}).get("by_horizon") or {}
                    if _bh:
                        st.caption("📐 " + str(_bt.get("generated_at", ""))[:10] + " 回测结果")
                        _cols = st.columns(max(1, len(_bh)))
                        for _i, (_h, _s) in enumerate(sorted(_bh.items(), key=lambda kv: int(kv[0]))):
                            with _cols[_i]:
                                st.metric(
                                    f"{_h}日 方向准确率",
                                    f"{_s['direction_accuracy']:.0%}",
                                    f"较基准 {_s['edge_vs_baseline']:+.0%}",
                                )
                                st.caption(
                                    f"朴素基准 {_s['baseline_accuracy']:.0%}　|　"
                                    f"平均误差 {_s['mae_pct']:.1f}%　|　样本 {_s['samples']}"
                                )
                    else:
                        st.info("还没有该股的回测数据。点下方「开始回测」，用历史数据检验模型的方向准确率（与朴素基准对照）。")

                    with st.expander("▶ 运行回测（评估模型历史准确率）", expanded=not _bh):
                        _bt_mode_label = st.radio(
                            "回测模式",
                            ["快档（仅 XGBoost，约 2 分钟）", "精档（全模型，约 25 分钟）"],
                            horizontal=True, key=f"bt_mode_{sym}",
                        )
                        _bt_mode = "accurate" if _bt_mode_label.startswith("精档") else "fast"
                        st.caption(
                            "口径：最近 6 个月、滚动 2 年训练窗、5 日步长（约 24 个评估点）。"
                            "每个评估点只用当时之前的数据现训现测，绝不使用未来数据；"
                            "方向准确率需超过基准才说明模型有效。"
                        )
                        if st.button("▶ 开始回测", key=f"bt_run_{sym}", type="primary") and run_backtest:
                            _pbar = st.progress(0, text="回测准备中…")

                            def _bt_cb(done, total, msg, _p=_pbar):
                                _p.progress(
                                    min(100, int(done / max(1, total) * 100)),
                                    text=f"回测中 {done}/{total}　{msg}",
                                )

                            try:
                                with st.spinner("回测进行中，请勿关闭或刷新页面…"):
                                    run_backtest(sym, mode=_bt_mode, progress=_bt_cb)
                                _pbar.empty()
                                st.success("✅ 回测完成，正在刷新结果…")
                                st.rerun()
                            except Exception as _bt_err:
                                _pbar.empty()
                                st.error(f"回测失败：{_bt_err}")

                # ── 风险管理 ──────────────────────────────────
                sent = r["sentiment"]
                sent_color = {"正面":"#EF5350","负面":"#22c55e"}.get(sent["label"],"#ADBAC7")
                pos_pct = sr.get("position_pct","—")
                pos_color = score_color(sr["total_score"])

                _mount_risk_expander_style()
                st.markdown('<div class="risk-expander-anchor"></div>', unsafe_allow_html=True)
                with st.expander("📌 风险管理建议", expanded=True):
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.markdown(cmet("建议止损价", f"¥{sr.get('stop_loss','—')}", "#EF5350", "跌破须止损"), unsafe_allow_html=True)
                    rc2.markdown(cmet("目标价位",   f"¥{sr.get('target_price','—')}", "#22c55e", "合理目标"), unsafe_allow_html=True)
                    rc3.markdown(cmet("建议仓位",   pos_pct, pos_color), unsafe_allow_html=True)
                    rc4.markdown(cmet("新闻情感", sent["label"], sent_color,
                                      f"{sent['score']:.1%} · 📰{sent.get('news_count',0)}条"), unsafe_allow_html=True)

                # ── 行情图 ────────────────────────────────────
                if "candle" in figs:
                    st.subheader("📉 行情走势 & 技术指标")
                    st.plotly_chart(figs["candle"], width='stretch', key=f"c_{sym}")
                elif r.get("_chart_error"):
                    st.error(f"📊 图表生成失败: {r['_chart_error']}")
                    st.caption("请查看启动 bat 窗口里的详细错误堆栈")

                # ── 预测图 ────────────────────────────────────
                if "predict" in figs:
                    st.subheader("🔮 价格预测走势")
                    st.plotly_chart(figs["predict"], width='stretch', key=f"p_{sym}")
                    pc1, pc2 = st.columns(2)
                    lp = r.get("long_pred")
                    if sp:
                        pct = sp["pct_change"]
                        arrow = "▲" if pct >= 0 else "▼"
                        pc1.markdown(cmet("短期（10天）", f"¥{sp['predictions'][-1]:.2f}",
                                          pct_color(pct), f"{arrow} {abs(pct):.2f}%", size="1.75em"),
                                     unsafe_allow_html=True)
                    if lp:
                        pct = lp["pct_change"]
                        arrow = "▲" if pct >= 0 else "▼"
                        pc2.markdown(cmet("中期（30天）", f"¥{lp['predictions'][-1]:.2f}",
                                          pct_color(pct), f"{arrow} {abs(pct):.2f}%", size="1.75em"),
                                     unsafe_allow_html=True)

                    # ── 回测验证准确率（简短摘要，详细面板见上方「模型回测」）──
                    try:
                        from analysis.backtest import get_latest_backtest, format_summary_line
                        _bt = get_latest_backtest(sym)
                        _bh = (_bt or {}).get("by_horizon") or {}
                        if _bh:
                            _parts = [format_summary_line(_bt, int(_h)) for _h in sorted(_bh, key=int)]
                            st.caption("📐 回测验证：　" + "　|　".join(_parts))
                    except Exception:
                        pass

                # ── 模型误差 (RMSE) ───────────────────────────
                sp = r.get("short_pred")
                lp = r.get("long_pred")
                
                # 获取各模型 RMSE
                all_rmse = {}
                
                if sp:
                    if sp.get("model_rmse"):
                        all_rmse.update({k: v for k, v in sp["model_rmse"].items() if v < float("inf")})
                
                if lp:
                    if lp.get("model_rmse"):
                        all_rmse.update({k: v for k, v in lp["model_rmse"].items() if v < float("inf")})
                
                # 快速模式只有 xgb_short 和 xgb_long
                if sp and sp.get("val_rmse_price"):
                    all_rmse["xgb_short"] = sp.get("val_rmse_price")
                if lp and lp.get("val_rmse_price"):
                    all_rmse["xgb_long"] = lp.get("val_rmse_price")
                
                if False and all_rmse:
                    with st.expander("📊 各模型验证误差 (RMSE)"):
                        rmse_order = [
                            ("lstm_short", "LSTM-S"),
                            ("lstm_long", "LSTM-L"),
                            ("xgb_short", "XGB-S"),
                            ("xgb_long", "XGB-L"),
                            ("prophet", "Prophet"),
                        ]
                        cols = st.columns(5)
                        for i, (key, display_name) in enumerate(rmse_order):
                            rmse_val = all_rmse.get(key)
                            with cols[i]:
                                if rmse_val is None:
                                    st.markdown(cmet(display_name, "--", "#ADBAC7"), unsafe_allow_html=True)
                                else:
                                    rmse_color = "#3fb950" if rmse_val < 2 else "#d29922" if rmse_val < 4 else "#f85149"
                                    st.markdown(cmet(display_name, f"{rmse_val:.2f}", rmse_color), unsafe_allow_html=True)
                        st.caption("RMSE 越低越好：<2 优秀 🟢 | 2-4 良好 🟡 | >4 偏差 🔴")

                # ── 历史评分 ──────────────────────────────────
                if "history" in figs:
                    st.subheader("📅 历史评分趋势")
                    st.plotly_chart(figs["history"], width='stretch', key=f"h_{sym}")

                # ── 技术信号 + 仪表 ───────────────────────────
                cs1, cs2 = st.columns([3, 2])
                with cs1:
                    st.subheader("📡 技术信号")
                    if "signals" in figs:
                        st.plotly_chart(figs["signals"], width='stretch', key=f"s_{sym}")
                    sigs = r["signals"].get("signals", [])
                    if sigs:
                        import pandas as pd
                        sig_df = pd.DataFrame([{
                            "指标": str(s["name"]),
                            "数值": str(s["value"]),
                            "信号": {"buy":"✅买入","sell":"❌卖出","neutral":"⚪中性"}[s["signal"]],
                            "说明": str(s["desc"]),
                        } for s in sigs])
                        st.dataframe(sig_df, hide_index=True, width='stretch')
                with cs2:
                    st.subheader("🎯 综合评分")
                    if "gauge" in figs:
                        st.plotly_chart(figs["gauge"], width='stretch', key=f"g_{sym}")

                # ── 基本面 ────────────────────────────────────
                cf1, cf2 = st.columns([1, 2])
                with cf1:
                    st.subheader("📋 基本面")
                    fund = r["fundamental"]
                    import pandas as pd
                    import pandas as pd
                    fd = pd.DataFrame([
                        ("行业",       str(fund.get("industry","—"))),
                        ("市盈率PE",   str(fmt(fund.get("pe"),".1f"))),
                        ("市净率PB",   str(fmt(fund.get("pb"),".2f"))),
                        ("ROE",        str(fmt(fund.get("roe"),".1f","%"))),
                        ("净利润增长", str(fmt(fund.get("profit_growth"),".1f","%"))),
                        ("营收增长",   str(fmt(fund.get("revenue_growth"),".1f","%"))),
                        ("股息率",     str(fmt(fund.get("dividend_yield"),".2f","%"))),
                        ("总市值",     str(fund.get("market_cap","—"))),
                    ], columns=["指标","数值"])
                    st.dataframe(fd, hide_index=True, width='stretch')
                with cf2:
                    st.subheader("🕸️ 基本面雷达")
                    if "radar" in figs:
                        st.plotly_chart(figs["radar"], width='stretch', key=f"r_{sym}")

                # ── 新闻 ──────────────────────────────────────
                news = r.get("news_list", [])
                if news:
                    st.subheader(f"📰 相关新闻（{len(news)} 条）")
                    news_rows = []
                    for idx, item in enumerate(news, start=1):
                        title = str(item.get("title", "") or "").strip()
                        if not title:
                            continue
                        content = str(item.get("content", "") or "").strip()
                        date_ = str(item.get("date", "") or "")[:10]
                        source = str(item.get("source", "") or "")
                        url_ = str(item.get("url", "") or "")
                        summary = ""
                        if content and content != title and len(content) > 10:
                            summary = content[:180] + ("..." if len(content) > 180 else "")
                        news_rows.append({
                            "序号": idx,
                            "日期": date_,
                            "来源": source,
                            "标题": title,
                            "摘要": summary,
                            "链接": url_,
                        })

                    if news_rows:
                        import pandas as pd
                        st.dataframe(
                            pd.DataFrame(news_rows),
                            hide_index=True,
                            width="stretch",
                            key=f"news_table_{sym}",
                        )
                        first_link = next((row["链接"] for row in news_rows if row.get("链接")), "")
                        if first_link:
                            st.link_button("打开第一条新闻原文", first_link, key=f"news_first_link_{sym}")

                # ── 回测 ──────────────────────────────────────
                bt = r.get("backtest")
                if bt:
                    st.subheader("⏱️ 策略回测")
                    wr  = bt.get("win_rate", 0)
                    ar  = bt.get("annual_return", 0)
                    md  = bt.get("max_drawdown", 0)
                    sr_ = bt.get("sharpe_ratio", 0)
                    er  = bt.get("excess_return", 0)
                    # 胜率/最大回撤/夏普：质量指标，绿=好 红=差
                    wr_c  = "#22c55e" if wr>=0.55  else ("#FFA726" if wr>=0.45  else "#f44336")
                    md_c  = "#22c55e" if abs(md)<=0.10 else ("#FFA726" if abs(md)<=0.20 else "#f44336")
                    sr_c  = "#22c55e" if sr_>=1.5  else ("#4CAF50" if sr_>=1    else ("#FFA726" if sr_>=0 else "#f44336"))
                    # 年化收益/超额收益：涨跌指标，A股 涨红跌绿
                    ar_c  = "#EF5350" if ar>0  else ("#22c55e" if ar<0  else "#ADBAC7")
                    er_c  = "#EF5350" if er>0  else ("#22c55e" if er<0  else "#ADBAC7")
                    bc1,bc2,bc3,bc4,bc5 = st.columns(5)
                    bc1.markdown(cmet("胜率",     f"{wr:.1%}", wr_c), unsafe_allow_html=True)
                    bc2.markdown(cmet("年化收益", f"{ar:.1%}", ar_c), unsafe_allow_html=True)
                    bc3.markdown(cmet("最大回撤", f"{md:.1%}", md_c), unsafe_allow_html=True)
                    bc4.markdown(cmet("夏普比率", f"{sr_:.2f}", sr_c), unsafe_allow_html=True)
                    bc5.markdown(cmet("超额收益", f"{er:+.1%}", er_c), unsafe_allow_html=True)
                    st.caption(f"区间：{bt.get('period','')}  ·  共 {bt.get('total_trades',0)} 次交易")

                # ── 支撑阻力 ──────────────────────────────────
                sr_lv = r.get("sr", {})
                if sr_lv:
                    st.subheader("📍 支撑位 & 阻力位")
                    sv1,sv2,sv3 = st.columns(3)
                    sv1.markdown(cmet("当前价", f"¥{sr_lv.get('close',0):.2f}", "#E6EDF3"), unsafe_allow_html=True)
                    sv2.markdown(cmet("阻力位", " / ".join(f"¥{x}" for x in sr_lv.get("resistance",[])[:3]), "#EF5350", "压力区间"), unsafe_allow_html=True)
                    sv3.markdown(cmet("支撑位", " / ".join(f"¥{x}" for x in sr_lv.get("support",[])[:3]), "#22c55e", "支撑区间"), unsafe_allow_html=True)

                st.divider()
                st.caption("⚠️ 风险提示：本报告由AI模型生成，仅供参考，不构成投资建议。")

    # ══ Tab 2: 多股看板 ══════════════════════════════════════
    with tab_board:
        st.subheader("🏆 多股对比看板")

        wl = st.session_state.watchlist
        if not wl:
            st.info("请先在左侧添加股票")
        else:
            if st.session_state.pop("_do_board", False):
                results = []
                pb = st.progress(0, "分析中...")
                for i, item in enumerate(wl):
                    pb.progress(int((i+1)/len(wl)*100), f"分析 {item['symbol']} {item.get('name','')}...")
                    results.append(run_quick(item["symbol"]))
                pb.empty()
                results.sort(key=lambda x: x.get("score_result",{}).get("total_score",0), reverse=True)
                st.session_state.board_results = results
                st.rerun()

            brs = st.session_state.board_results
            if not brs:
                st.info("点击左侧「📊 多股对比看板」开始分析")
            else:
                import pandas as pd
                rows = []
                for i, br in enumerate(brs):
                    if "error" in br:
                        rows.append({"排名":i+1,"代码":br["symbol"],"名称":"错误",
                                     "综合评分":"—","评级":"—","技术":"—","基本面":"—",
                                     "情感":"—","当前价":"—","10天":"—","30天":"—"})
                        continue
                    sr_ = br.get("score_result", {})
                    sp_ = br.get("short_pred")
                    lp_ = br.get("long_pred")
                    rows.append({
                        "排名":   str(i+1),
                        "代码":   str(br["symbol"]),
                        "名称":   str(br["name"]),
                        "综合评分": f"{sr_.get('total_score', 0):.1f}",
                        "评级":   str(sr_.get("rating","—")),
                        "技术":   f"{sr_.get('technical_score', 0):.1f}",
                        "基本面": f"{sr_.get('fundamental_score', 0):.1f}",
                        "情感":   f"{sr_.get('sentiment_score', 0):.1f}",
                        "当前价": f"¥{br.get('last_price', 0):.2f}",
                        "10天":   (f"{'▲' if sp_['pct_change']>=0 else '▼'}{abs(sp_['pct_change']):.1f}%" if sp_ else "—"),
                        "30天":   (f"{'▲' if lp_['pct_change']>=0 else '▼'}{abs(lp_['pct_change']):.1f}%" if lp_ else "—"),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
                             height=min(420, 55+len(rows)*50))

                # 雷达对比图
                try:
                    import plotly.graph_objects as go
                    cats = ["技术面","基本面","情感面","预测面","技术面"]
                    fig_rd = go.Figure()
                    for br in brs:
                        if "error" in br: continue
                        s_ = br.get("score_result",{})
                        fig_rd.add_trace(go.Scatterpolar(
                            r=[s_.get("technical_score",50), s_.get("fundamental_score",50),
                               s_.get("sentiment_score",50), s_.get("prediction_score",50),
                               s_.get("technical_score",50)],
                            theta=cats, fill="toself",
                            name=f"{br['symbol']} {br['name']}",
                        ))
                    fig_rd.update_layout(
                        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                        font=dict(color="#C9D1D9"),
                        polar=dict(bgcolor="#161B22",
                                   radialaxis=dict(range=[0,100], showticklabels=False),
                                   angularaxis=dict(gridcolor="#21262D")),
                        title="各股维度对比", height=420,
                    )
                    st.plotly_chart(fig_rd, width='stretch', key="board_radar")
                except: pass

    # ══ Tab 3: 历史评分 ══════════════════════════════════════
    with tab_history:
        st.subheader("📅 历史评分记录")

        wl = st.session_state.watchlist
        if not wl:
            st.info("请先添加股票")
        else:
            syms  = [w["symbol"] for w in wl]
            names = {w["symbol"]: w.get("name", w["symbol"]) for w in wl}
            sel_h = st.selectbox("选择股票",  syms,
                                  format_func=lambda s: f"{s}  {names.get(s,'')}",
                                  key="hist_sel")
            if sel_h:
                try:
                    from data.score_history import load_history
                    records = load_history(sel_h)
                except: records = []

                if not records:
                    st.info("暂无记录，运行「分析当前股票」后自动保存")
                else:
                    from visualization.charts import create_score_history_chart
                    st.plotly_chart(
                        create_score_history_chart(records, names.get(sel_h, sel_h)),
                        width='stretch', key="hist_chart",
                    )

                    import pandas as pd
                    hdf = pd.DataFrame([{
                        "日期":    str(r.get("date",""))[:10] if r.get("date") else "—",
                        "综合评分": f"{r.get('total_score', 0):.1f}" if r.get("total_score") is not None else "—",
                        "技术面":  f"{r.get('technical_score', 0):.1f}" if r.get("technical_score") is not None else "—",
                        "基本面":  f"{r.get('fundamental_score', 0):.1f}" if r.get("fundamental_score") is not None else "—",
                        "情感面":  f"{r.get('sentiment_score', 0):.1f}" if r.get("sentiment_score") is not None else "—",
                        "10天预测":(f"{'▲' if r['short_pct']>=0 else '▼'}{abs(r['short_pct']):.1f}%"
                                    if r.get("short_pct") is not None else "—"),
                        "评级":    str(r.get("rating", "—")),
                    } for r in reversed(records)])
                    st.dataframe(hdf, hide_index=True, width='stretch')

                    scores = [r["total_score"] for r in records if r.get("total_score")]
                    if len(scores) > 1:
                        m1,m2,m3,m4 = st.columns(4)
                        m1.metric("最高评分", f"{max(scores):.1f}")
                        m2.metric("最低评分", f"{min(scores):.1f}")
                        m3.metric("平均评分", f"{sum(scores)/len(scores):.1f}")
                        m4.metric("记录总数", len(scores))

    # ══ Tab 4: 在线学习 ══════════════════════════════════════
    with tab_online:
        st.subheader('🤖 在线学习与模型管理')

        if not ONLINE_LEARNING_ENABLED:
            st.info('在线学习功能尚未启用，请先在左侧开启“实现在线学习”。')
        else:
            wl = st.session_state.watchlist
            task_status = _query_scheduled_task()
            latest_learning = _latest_learning_report_status()
            bg_runtime = _background_learning_runtime_status()

            st.markdown('#### 📌 后台学习状态')
            s1, s2, s3, s4 = st.columns(4)
            s1.metric('计划任务', '已注册' if task_status.get('registered') else '未注册', task_status.get('status') or '')
            s2.metric('下次执行', task_status.get('next_run') or '--')
            s3.metric('最近学习', latest_learning.get('update_time') or '--', latest_learning.get('symbol') or '')
            s4.metric('最近更新模型', str(latest_learning.get('updated_count', '--')), f'告警 {latest_learning.get("alert_count", 0)}')
            if latest_learning.get('path'):
                st.caption(f'最近报告：{latest_learning["path"]}')

            if bg_runtime:
                if bg_runtime.get('state') == 'running':
                    st.info(f'后台学习进行中，已完成 {bg_runtime.get("completed_count", 0)}/{bg_runtime.get("total_count", 0)} 只；PID={bg_runtime.get("pid")}')
                    progress_value = 0.0
                    total_count = bg_runtime.get('total_count', 0) or 0
                    if total_count:
                        progress_value = bg_runtime.get('completed_count', 0) / total_count
                    st.progress(
                        progress_value,
                        text=(
                            f'最近完成：{bg_runtime.get("latest_symbol")} {bg_runtime.get("latest_update_time")}'
                            if bg_runtime.get('latest_symbol')
                            else '后台学习已启动，等待第一个结果...'
                        ),
                    )
                elif bg_runtime.get('state') == 'finished':
                    st.success(f'最近一次后台学习已经完成，共 {bg_runtime.get("completed_count", 0)}/{bg_runtime.get("total_count", 0)} 只')
                elif bg_runtime.get('state') == 'stopped':
                    st.warning(f'后台学习进程已结束，当前完成 {bg_runtime.get("completed_count", 0)}/{bg_runtime.get("total_count", 0)} 只')

            st.markdown('#### ⚙️ 在线学习控制')
            c1, c2, c3, c4, c5, c6 = st.columns(6)

            with c1:
                if st.button('🔄 立即执行全量更新', width='stretch', key='tab_online_run_full_update'):
                    try:
                        from models.online_learner import run_daily_update_all
                        symbols = [w['symbol'] for w in wl]
                        with st.spinner('正在执行在线学习...'):
                            reports = run_daily_update_all(symbols, workers=1)
                        st.success('在线学习完成')
                        summary_rows = []
                        for symbol, report in reports.items():
                            summary_rows.append({
                                '股票': symbol,
                                '更新模型数': len(report.get('models_updated', []) or []),
                                '跳过模型数': len(report.get('models_skipped', []) or []),
                                '告警数': len(report.get('alerts', []) or []),
                            })
                        if summary_rows:
                            st.dataframe(summary_rows, width='stretch', hide_index=True)
                    except Exception as e:
                        st.error(f'执行在线学习失败: {e}')

            with c2:
                if st.button('🚀 后台执行一次', width='stretch', key='tab_online_run_background_once'):
                    try:
                        symbols = [w['symbol'] for w in wl]
                        if not symbols:
                            st.warning('请先添加自选股后再启动后台学习')
                        else:
                            launch_info = _launch_background_learning(symbols)
                            if launch_info.get('already_running'):
                                st.warning(f'后台自学习已在运行，进程 PID: {launch_info["pid"]}')
                            else:
                                st.success(f'后台自学习已启动，进程 PID: {launch_info["pid"]}')
                    except Exception as e:
                        st.error(f'启动后台自学习失败: {e}')

            with c3:
                if st.button('🗓 注册每日后台学习', width='stretch', key='tab_online_register_daily_task'):
                    try:
                        symbols = [w['symbol'] for w in wl]
                        if not symbols:
                            st.warning('请先添加自选股后再注册后台学习')
                        else:
                            register_info = _register_background_learning(symbols)
                            st.success('已注册每日后台学习计划任务')
                            st.caption(register_info['helper']['register'])
                    except Exception as e:
                        st.error(f'注册每日后台学习失败: {e}')

            with c4:
                if st.button('🛑 注销每日后台学习', width='stretch', key='tab_online_unregister_daily_task'):
                    try:
                        _unregister_background_learning()
                        st.success('已注销每日后台学习计划任务')
                    except Exception as e:
                        st.error(f'注销每日后台学习失败: {e}')

            with c5:
                if st.button('📥 下载性能报告', width='stretch', key='tab_online_download_report'):
                    st.info('性能报告已保存在 performance_logs 目录中')

            with c6:
                if st.button('🧹 清理旧模型', width='stretch', key='tab_online_cleanup_models'):
                    try:
                        symbols = [w['symbol'] for w in wl]
                        if not symbols:
                            st.warning('请先添加自选股后再清理模型')
                        else:
                            from models.online_learner import OnlineLearner
                            cleanup = OnlineLearner().cleanup_model_versions(symbols)
                            st.success(
                                f"已清理 {cleanup['removed_count']} 个旧模型，释放约 {cleanup['freed_mb']} MB；"
                                f"当前策略：每种模型保留最近 {cleanup['keep_recent']} 个 + 最优 {cleanup['keep_best']} 个"
                            )
                    except Exception as e:
                        st.error(f'清理旧模型失败: {e}')

            with st.expander('查看模型明细', expanded=False):
                if not wl:
                    st.info('请先添加股票')
                else:
                    syms = [w['symbol'] for w in wl]
                    names = {w['symbol']: w.get('name', w['symbol']) for w in wl}
                    sel_online = st.selectbox(
                        '选择股票查看学习状态',
                        syms,
                        format_func=lambda s: f'{s}  {names.get(s, "")}',
                        key='online_sel_slim',
                    )
                    if sel_online:
                        try:
                            from models.online_learner import OnlineLearner
                            learner = OnlineLearner()
                            controls = learner.get_model_control(sel_online)
                            latest_report = learner.get_latest_report(sel_online)
                            if latest_report:
                                st.caption(
                                    f"最近报告：{latest_report.get('symbol', sel_online)}"
                                    f" · {latest_report.get('update_time', '未知时间')}"
                                )
                                updated_rows = latest_report.get('models_updated', []) or []
                                skipped_rows = latest_report.get('models_skipped', []) or []
                                alert_rows = latest_report.get('alerts', []) or []

                                r1, r2, r3 = st.columns(3)
                                for col, label, value in (
                                    (r1, '更新模型', len(updated_rows)),
                                    (r2, '跳过模型', len(skipped_rows)),
                                    (r3, '警告数量', len(alert_rows)),
                                ):
                                    col.markdown(
                                        f"<div class='score-card' style='text-align:left'>"
                                        f"<div class='lbl'>{label}</div>"
                                        f"<div class='val'>{value}</div>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                                if updated_rows:
                                    st.markdown('##### 更新模型')
                                    st.dataframe(updated_rows, width='stretch', hide_index=True)
                                if skipped_rows:
                                    st.markdown('##### 跳过模型')
                                    st.dataframe(skipped_rows, width='stretch', hide_index=True)
                                if alert_rows:
                                    st.markdown('##### 告警')
                                    st.dataframe(alert_rows, width='stretch', hide_index=True)
                            if controls:
                                control_rows = []
                                for model_key, state in controls.items():
                                    control_rows.append({
                                        'model': model_key,
                                        'status': state.get('status'),
                                        'disabled': state.get('disabled'),
                                        'downweight_factor': state.get('downweight_factor', 1.0),
                                        'samples': state.get('count', 0),
                                    })
                                st.markdown('##### 模型控制状态')
                                st.dataframe(control_rows, width='stretch', hide_index=True)
                        except Exception as e:
                            st.error(f'加载模型明细失败: {e}')

