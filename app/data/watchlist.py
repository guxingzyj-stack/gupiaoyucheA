import json
import os
from typing import Dict, List

import config


DEFAULT_WATCHLIST: List[Dict[str, str]] = []


def load_watchlist() -> List[Dict[str, str]]:
    if not os.path.exists(config.WATCHLIST_FILE):
        return list(DEFAULT_WATCHLIST)
    try:
        with open(config.WATCHLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return list(DEFAULT_WATCHLIST)


def save_watchlist(watchlist: List[Dict[str, str]]) -> None:
    # 相对路径文件名时 dirname 返回空串，makedirs("") 会抛错
    dir_path = os.path.dirname(config.WATCHLIST_FILE)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(config.WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)
