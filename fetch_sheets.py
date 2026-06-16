"""
Google Sheets → CSV export script.

Usage:
    python -X utf8 fetch_sheets.py

Exports:
    data/cdr.csv
    data/staffing.csv
    data/ring_config.csv
"""

import sys
import os
import json
import urllib.request
import urllib.parse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ---- paths -----------------------------------------------------------

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Sheets設定 ------------------------------------------------------

SHEETS = [
    {
        "id":    "1dUgQ_VObJGb7LYXARUyzKr5dK-qC3qp9MNhLyw44nsM",
        "sheet": "CDR",
        "out":   DATA_DIR / "cdr.csv",
        "label": "CDR",
    },
    {
        "id":    "1dUgQ_VObJGb7LYXARUyzKr5dK-qC3qp9MNhLyw44nsM",
        "sheet": "稼働時間",
        "out":   DATA_DIR / "staffing.csv",
        "label": "稼働時間",
    },
    {
        "id":    "1dUgQ_VObJGb7LYXARUyzKr5dK-qC3qp9MNhLyw44nsM",
        "sheet": "内線種別",
        "out":   DATA_DIR / "ext_type.csv",
        "label": "内線種別",
    },
    {
        "id":    "16YMJP3gRn62F6Ktn52AHsxAR6sj5hJU_h8bO3CufvBo",
        "sheet": "エージェント鳴動設定表",
        "out":   DATA_DIR / "ring_config.csv",
        "label": "鳴動設定表",
    },
]

# ---- export URL ------------------------------------------------------

def export_url(sheet_id: str, sheet_name: str) -> str:
    """Google SheetsのCSVエクスポートURLを生成する。"""
    encoded = urllib.parse.quote(sheet_name)
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={encoded}"
    )

# ---- fetch -----------------------------------------------------------

def fetch_csv(url: str, out_path: Path, label: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        out_path.write_text(content, encoding="utf-8")
        rows = len(content.strip().splitlines()) - 1
        print(f"  ✅ {label}: {rows}行 → {out_path.name}")
    except Exception as e:
        print(f"  ❌ {label}: 取得失敗 - {e}", file=sys.stderr)
        sys.exit(1)

# ---- main ------------------------------------------------------------

def main() -> None:
    print("📥 Google Sheetsからデータ取得中...")
    for s in SHEETS:
        url = export_url(s["id"], s["sheet"])
        fetch_csv(url, s["out"], s["label"])
    print("✅ 全シート取得完了")

if __name__ == "__main__":
    main()
