"""
Chatwork notification script for call center daily analysis.

Usage:
    python call_center/notify.py [YYYY-MM-DD]

    Date defaults to today if omitted.

Required (.env or env vars):
    CHATWORK_API_TOKEN      Chatwork API token
    CHATWORK_ADMIN_ROOM_ID  Admin room ID

Optional env vars:
    REPORTS_DIR  Path to reports folder (default: call_center/reports/)
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path


# ---- constants -------------------------------------------------------

CHATWORK_API_BASE = "https://api.chatwork.com/v2"
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"
DOTENV_PATH = PROJECT_ROOT / ".env"

WEEKDAYS = ["月", "火", "水", "木", "金", "土", "日"]


def _dw(s: str) -> int:
    """Display width: non-ASCII (CJK, emoji, etc.) = 2, ASCII = 1."""
    return sum(2 if ord(c) > 0x7F else 1 for c in s)


def _rjust_dw(s: str, width: int) -> str:
    """Right-justify string to display width using ASCII spaces for padding."""
    return ' ' * max(0, width - _dw(s)) + s


def _fmt_secs(secs) -> str:
    """Format seconds to mm:ss string, or '-' if None."""
    if secs is None:
        return "-"
    s = int(secs)
    return f"{s // 60}分{s % 60:02d}秒"


# ---- env / config ----------------------------------------------------

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def load_env() -> dict:
    token = os.environ.get("CHATWORK_API_TOKEN", "")
    admin_room = os.environ.get("CHATWORK_ADMIN_ROOM_ID", "")
    if not token:
        raise EnvironmentError("CHATWORK_API_TOKEN が設定されていません")
    if not admin_room:
        raise EnvironmentError("CHATWORK_ADMIN_ROOM_ID が設定されていません")
    return {
        "token": token,
        "admin_room": admin_room,
        "reports_dir": Path(os.environ.get("REPORTS_DIR", DEFAULT_REPORTS_DIR)),
    }


# ---- report loading --------------------------------------------------

def load_report_json(target_date: date, reports_dir: Path) -> dict:
    path = reports_dir / "daily" / f"{target_date}.html"
    if not path.exists():
        raise FileNotFoundError(f"レポートが見つかりません: {path}")
    raw = path.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="rdata">(.*?)</script>',
                  raw, re.DOTALL)
    if not m:
        raise ValueError(f"レポートにrdata JSONが見つかりません: {path}")
    return json.loads(m.group(1))


# ---- report parsing --------------------------------------------------

def _extract_diff_str(cell: str) -> str:
    """Extract short display string from a _diff_str()-formatted cell."""
    cell = cell.strip()
    if not cell or cell in ('-', '—', '－'):
        return "—"
    m = re.search(r"\(([+\-][\d.]+%)\)", cell)
    if m:
        return m.group(1)
    m = re.search(r"([+\-±][\d.]+pt)", cell)
    if m:
        return m.group(1)
    return "—"


def _rate_flag(rate: float) -> str:
    if rate >= 90:
        return "🟢"
    if rate >= 80:
        return "🟡"
    return "🔴"


def extract_kpi(rdata: dict) -> dict:
    diffs = rdata.get("kpi_diffs", {})
    rate  = rdata.get("rate")
    work_hrs = rdata.get("work_hrs")
    avg_cph  = rdata.get("avg_cph")
    avg_talk = rdata.get("avg_talk_secs")

    rate_val  = f"{rate}%" if rate is not None else "—"
    rate_flag = _rate_flag(float(rate)) if rate is not None else ""
    work_val  = f"{work_hrs:.1f}h" if work_hrs is not None else "—"
    cph_val   = str(avg_cph) if avg_cph is not None else "—"
    talk_val  = _fmt_secs(avg_talk)

    return {
        "total":       f"{rdata.get('total', '—')}件",
        "got":         f"{rdata.get('got', '—')}件",
        "missed":      f"{rdata.get('missed', '—')}件",
        "rate":        rate_val,
        "rate_flag":   rate_flag,
        "work_hrs":    work_val,
        "cph":         cph_val,
        "talk_time":   talk_val,
        "total_day":   _extract_diff_str(diffs.get("total_day", "-")),
        "total_week":  _extract_diff_str(diffs.get("total_week", "-")),
        "got_day":     _extract_diff_str(diffs.get("got_day", "-")),
        "got_week":    _extract_diff_str(diffs.get("got_week", "-")),
        "missed_day":  _extract_diff_str(diffs.get("missed_day", "-")),
        "missed_week": _extract_diff_str(diffs.get("missed_week", "-")),
        "rate_day":    _extract_diff_str(diffs.get("rate_day", "-")),
        "rate_week":   _extract_diff_str(diffs.get("rate_week", "-")),
        "work_day":    _extract_diff_str(diffs.get("work_day", "-")),
        "work_week":   _extract_diff_str(diffs.get("work_week", "-")),
        "cph_day":     _extract_diff_str(diffs.get("cph_day", "-")),
        "cph_week":    _extract_diff_str(diffs.get("cph_week", "-")),
        "talk_day":    _extract_diff_str(diffs.get("talk_day", "-")),
        "talk_week":   _extract_diff_str(diffs.get("talk_week", "-")),
    }


def _format_ext_row(rank: int, row: dict) -> str:
    flag = _rate_flag(row["rate"])
    return (
        f"{rank}位　{row['ext']}　{row['name']}　"
        f"着信 {row['total']:>3}件　応答 {row['got']:>3}件　未応答 {row['missed']:>3}件　"
        f"応答率 {row['rate']:>5}%　{flag}"
    )


def _format_op_cph_row(rank: int, row: dict) -> str:
    rate = row.get("rate")
    flag = _rate_flag(float(rate)) if rate is not None else ""
    rate_str = f"{rate:>5}%" if rate is not None else "    -"
    cph_str = str(row["cph"]) if row.get("cph") is not None else "—"
    return (
        f"{rank}位　{row['name']}（{row['zoiper']}）　"
        f"CPH {_rjust_dw(cph_str, 4)}　着信 {row['total']:>3}件　応答 {row['got']:>3}件　"
        f"未応答 {row['missed']:>3}件　応答率 {rate_str}　{flag}"
    )


# ---- message building ------------------------------------------------

def _format_factors(factors: dict) -> list[str]:
    lines = []
    for label in ["需要", "供給", "生産性"]:
        items = factors.get(label, [])
        if items:
            for item in items:
                lines.append(f"  {label}要因：{item}")
        else:
            lines.append(f"  {label}要因：確認できません")
    return lines


def build_message(target_date: date, rdata: dict) -> str:
    weekday  = WEEKDAYS[target_date.weekday()]
    date_str = f"{target_date.year}/{target_date.month:02d}/{target_date.day:02d}"

    kpi     = extract_kpi(rdata)
    factors = rdata.get("factors", {})

    v = kpi
    lines = [
        f"[info][title]📊 コール日次分析 {date_str}（{weekday}）[/title]",
        "【📈 KPIサマリー】",
        f"着信数　　　：{_rjust_dw(v['total'], 7)}  前日比 {_rjust_dw(v['total_day'], 7)}  前週比 {_rjust_dw(v['total_week'], 7)}",
        f"応答数　　　：{_rjust_dw(v['got'], 7)}  前日比 {_rjust_dw(v['got_day'], 7)}  前週比 {_rjust_dw(v['got_week'], 7)}",
        f"未応答数　　：{_rjust_dw(v['missed'], 7)}  前日比 {_rjust_dw(v['missed_day'], 7)}  前週比 {_rjust_dw(v['missed_week'], 7)}",
        f"応答率　　　：{_rjust_dw(v['rate'], 7)}  前日比 {_rjust_dw(v['rate_day'], 7)}  前週比 {_rjust_dw(v['rate_week'], 7)}  {v['rate_flag']}",
        f"合計稼働時間：{_rjust_dw(v['work_hrs'], 7)}  前日比 {_rjust_dw(v['work_day'], 7)}  前週比 {_rjust_dw(v['work_week'], 7)}",
        f"平均CPH　　 ：{_rjust_dw(v['cph'], 7)}  前日比 {_rjust_dw(v['cph_day'], 7)}  前週比 {_rjust_dw(v['cph_week'], 7)}",
        f"平均通話時間：{_rjust_dw(v['talk_time'], 7)}  前日比 {_rjust_dw(v['talk_day'], 7)}  前週比 {_rjust_dw(v['talk_week'], 7)}",
        "",
        "【🔍 要因分析】",
        *_format_factors(factors),
        "",
        "【🌐 ダッシュボード】",
        "https://anabuki-bonz.github.io/call-analysis/call_center/reports/dashboard.html",
        "[/info]",
    ]
    return "\n".join(lines)


# ---- Chatwork API ----------------------------------------------------

def send_message(token: str, room_id: str, message: str) -> None:
    url = f"{CHATWORK_API_BASE}/rooms/{room_id}/messages"
    data = urllib.parse.urlencode({"body": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"X-ChatWorkToken": token},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"Chatwork API エラー: HTTP {resp.status}")


# ---- main ------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    print("⏳ GitHub Pages反映待ち中（120秒）...", flush=True)
    time.sleep(120)
    load_dotenv(DOTENV_PATH)

    if len(sys.argv) >= 2:
        try:
            target_date = date.fromisoformat(sys.argv[1])
        except ValueError:
            print(f"日付形式エラー: {sys.argv[1]}（例: 2026-05-22）", file=sys.stderr)
            sys.exit(1)
    else:
        target_date = date.today()

    try:
        env = load_env()
    except EnvironmentError as e:
        print(f"設定エラー: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        rdata = load_report_json(target_date, env["reports_dir"])
    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)

    message = build_message(target_date, rdata)

    try:
        send_message(env["token"], env["admin_room"], message)
        print(f"✅ Chatwork通知送信完了（{target_date}）")
    except Exception as e:
        print(f"❌ 送信失敗: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
