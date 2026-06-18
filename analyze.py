"""
Call center daily analysis script.

Usage:
    python -X utf8 analyze.py [YYYY-MM-DD]

Output:
    reports/daily/YYYY-MM-DD.html
    index.html (dashboard)

Note: OP-level call counts are derived from 取得Zoiper / 放棄Zoiper columns.
      Calls where both are NaN cannot be attributed to a specific OP.
"""

import json
import math
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR = REPORTS_DIR / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)


def resolve_date(arg: str | None) -> date:
    if arg:
        return date.fromisoformat(arg)
    return date.today() - timedelta(days=1)


def prev_business_day(d: date) -> date:
    return d - timedelta(days=3 if d.weekday() == 0 else 1)


CDR_COLS = ["入電判定", "入電結果", "日付", "曜日", "時刻", "内線番号", "内線種別", "案件名", "取得Zoiper", "放棄Zoiper"]
STF_COLS = ["Div.", "報告日", "Zoiper", "OP名", "勤務時間", "出勤時間", "退勤時間", "休憩入り", "休憩戻り"]
EXT_COLS = ["種別名称一覧 No.", "Div."]


def check_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ {label}: 列が見つかりません: {missing}", file=sys.stderr)
        sys.exit(1)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "cdr.csv":       DATA_DIR / "cdr.csv",
        "staffing.csv":  DATA_DIR / "staffing.csv",
        "ext_type.csv":  DATA_DIR / "ext_type.csv",
        "ring_config.csv": DATA_DIR / "ring_config.csv",
    }
    for name, p in paths.items():
        if not p.exists():
            print(f"❌ データファイルが見つかりません: {p}", file=sys.stderr)
            sys.exit(1)

    cdr  = pd.read_csv(paths["cdr.csv"],       encoding="utf-8-sig")
    stf  = pd.read_csv(paths["staffing.csv"],  encoding="utf-8-sig")
    ext  = pd.read_csv(paths["ext_type.csv"],  encoding="utf-8-sig")
    ring = pd.read_csv(paths["ring_config.csv"], encoding="utf-8-sig")

    check_columns(cdr, CDR_COLS, "cdr.csv")
    check_columns(stf, STF_COLS, "staffing.csv")
    check_columns(ext, EXT_COLS, "ext_type.csv")

    stf["報告日"] = pd.to_datetime(stf["報告日"]).dt.strftime("%Y/%m/%d")
    return cdr, stf, ext, ring


# ---- OP status -------------------------------------------------------

def get_op_status(op_row: pd.Series, hour: int) -> str:
    try:
        start = float(op_row["出勤時間"])
        end   = float(op_row["退勤時間"])
    except (ValueError, KeyError):
        return "off"
    in_shift = (start <= hour < end) if end > start else (hour >= start or hour < end)
    if not in_shift:
        return "off"
    b_in  = op_row.get("休憩入り")
    b_out = op_row.get("休憩戻り")
    if pd.notna(b_in) and pd.notna(b_out):
        b_in, b_out = float(b_in), float(b_out)
        in_break = (b_in <= hour < b_out) if b_out > b_in else (hour >= b_in or hour < b_out)
        if in_break:
            return "break"
    return "active"


def get_in_ops_by_status(stf: pd.DataFrame, date_str: str, hour: int) -> tuple[dict, dict]:
    day = stf[(stf["Div."] == "IN") & (stf["報告日"].astype(str).str.contains(date_str))]
    active, on_break = {}, {}
    for _, row in day.iterrows():
        status = get_op_status(row, hour)
        zoiper = str(int(row["Zoiper"])) if pd.notna(row.get("Zoiper")) else None
        if zoiper is None:
            continue
        if status == "active":
            active[zoiper] = str(row.get("OP名", ""))
        elif status == "break":
            on_break[zoiper] = str(row.get("OP名", ""))
    return active, on_break


# ---- ring config -----------------------------------------------------

def build_capable_map(ring: pd.DataFrame) -> dict[str, set[str]]:
    capable_map: dict[str, set[str]] = {}
    header_row = ring.iloc[0]
    for col in ring.columns[1:]:
        type_num = str(header_row[col]).strip()
        zoipers: set[str] = set()
        for _, row in ring.iloc[1:].iterrows():
            if str(row[col]).strip() == "○":
                zoipers.add(str(row.iloc[0]).strip())
        capable_map[type_num] = zoipers
    return capable_map


# ---- miss classification ---------------------------------------------

def classify_miss(active: dict, on_break: dict,
                  ext_type: str | None, capable_map: dict[str, set[str]],
                  abandon_zoiper: bool = False) -> str:
    if not active and not on_break:
        return "(4) 稼働なし"
    if abandon_zoiper:
        return "(2) 混雑"
    if not active:
        if ext_type and ext_type in capable_map:
            if {z for z in on_break if z in capable_map[ext_type]}:
                return "(1) 休憩中"
        return "(4) 稼働なし"
    if ext_type is None or ext_type not in capable_map:
        return "(3) 範囲外"
    capable        = capable_map[ext_type]
    active_capable = {z for z in active   if z in capable}
    break_capable  = {z for z in on_break if z in capable}
    if not active_capable and not break_capable:
        return "(3) 範囲外"
    if not active_capable and break_capable:
        return "(1) 休憩中"
    return "(3) 範囲外"


# ---- helpers ---------------------------------------------------------

def normalize_type(val) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    m = re.match(r"\d{4}-(\d{2})-(\d{2})", s)
    if m:
        return f"{int(m.group(1))}-{int(m.group(2))}"
    return s


def get_ext_type_label(name: str) -> str:
    name = str(name).strip()
    if name.startswith("[A]") or name.startswith("［A］"):
        return "A"
    if name.startswith("[B]") or name.startswith("［B］"):
        return "B"
    return "C"


def to_zoiper_str(v) -> str | None:
    try:
        if pd.isna(v):
            return None
        return str(int(float(v)))
    except Exception:
        return None


def to_zoiper_list(v) -> list[str]:
    """Parse 放棄Zoiper cell which may contain comma-separated multiple values."""
    if pd.isna(v) if not isinstance(v, str) else not v.strip():
        return []
    result = []
    for part in str(v).split(","):
        part = part.strip()
        try:
            result.append(str(int(float(part))))
        except Exception:
            pass
    return result


def rate_flag(r) -> str:
    if r is None:
        return ""
    if r >= 90:
        return "🟢"
    if r >= 80:
        return "🟡"
    return "🔴"


def rate_class(r) -> str:
    if r is None:
        return ""
    if r >= 90:
        return "green"
    if r >= 80:
        return "yellow"
    return "red"


def kpi_label(rate: float | None) -> str:
    if rate is None:
        return "-"
    if rate >= 98:  return "S"
    if rate >= 95:  return "A"
    if rate >= 90:  return "B"
    if rate >= 85:  return "C"
    return "D"


def kpi_badge_html(rate: float | None) -> str:
    label = kpi_label(rate)
    if label == "-":
        return "<td>-</td>"
    return f'<td><span class="kpi-badge kpi-{label.lower()}">{label}</span></td>'


def fmt_seconds(secs) -> str:
    if secs is None or (isinstance(secs, float) and math.isnan(secs)):
        return "-"
    m, s = divmod(int(secs), 60)
    return f"{m}分{s:02d}秒"


def _diff_str(new_val, old_val, unit: str = "", is_rate: bool = False,
              use_float: bool = False) -> str:
    """Format a comparison cell. is_rate=True shows pt, else count+ratio."""
    if old_val is None or new_val is None:
        return "-"
    diff = new_val - old_val
    if diff == 0:
        return "±0pt" if is_rate else "±0"
    sign = "+" if diff > 0 else ""
    if is_rate:
        return f"{sign}{round(diff, 1)}pt"
    if old_val == 0:
        return "-"
    pct  = round(diff / abs(old_val) * 100, 1)
    psign = "+" if pct >= 0 else ""
    if use_float:
        diff_disp = f"{sign}{diff:.1f}{unit}"
    else:
        diff_disp = f"{sign}{int(round(diff))}{unit}"
    return f"{diff_disp}({psign}{pct}%)"


# ---- basic stats for any date (for comparison rows) ------------------

def _prev_month_weekday_avg(
    cdr: pd.DataFrame, stf: pd.DataFrame,
    target_date: date, weekday_label: str, in_ext_types: set
) -> dict | None:
    """前月の同曜日（祝日除外）の平均stats。weekday_labelが「祝」の場合は祝日同士で平均。"""
    import calendar as _cal
    pm_year  = target_date.year if target_date.month > 1 else target_date.year - 1
    pm_month = target_date.month - 1 if target_date.month > 1 else 12
    days_in_pm = _cal.monthrange(pm_year, pm_month)[1]

    stats_list: list[dict] = []
    for day in range(1, days_in_pm + 1):
        d = date(pm_year, pm_month, day)
        date_csv = str(d).replace("-", "/")
        day_in = cdr[
            (cdr["入電判定"] == True) &
            (cdr["日付"].astype(str).str.contains(date_csv, regex=False)) &
            (cdr["内線種別"].isin(in_ext_types))
        ]
        if day_in.empty:
            continue
        wd = str(day_in["曜日"].iloc[0])
        if weekday_label == "祝":
            if wd != "祝":
                continue
        else:
            if wd != weekday_label or wd == "祝":
                continue
        s = _day_stats(cdr, stf, d, in_ext_types)
        if s:
            stats_list.append(s)

    if not stats_list:
        return None
    n = len(stats_list)
    keys = ["total", "got", "missed", "rate", "work_hrs", "avg_cph", "avg_talk"]
    avg: dict = {}
    for k in keys:
        vals = [s[k] for s in stats_list if s.get(k) is not None]
        avg[k] = round(sum(vals) / len(vals), 1) if vals else None
    avg["_n"] = n
    return avg


def _day_stats(cdr: pd.DataFrame, stf: pd.DataFrame,
               d: date, in_ext_types: set) -> dict | None:
    date_csv = str(d).replace("-", "/")
    day_in = cdr[
        (cdr["入電判定"] == True) &
        (cdr["日付"].astype(str).str.contains(date_csv)) &
        (cdr["内線種別"].isin(in_ext_types))
    ]
    if day_in.empty:
        return None
    total  = len(day_in)
    got    = int((day_in["入電結果"] == True).sum())
    missed = total - got
    rate   = round(got / total * 100, 1)
    day_stf = stf[(stf["Div."] == "IN") & (stf["報告日"].astype(str).str.contains(date_csv))]
    ops      = len(day_stf)
    work_hrs = float(day_stf["勤務時間"].sum()) if not day_stf.empty else 0.0
    avg_cph  = round(got / work_hrs, 1) if work_hrs > 0 else None
    op_per   = round(total / ops, 1) if ops > 0 else None
    needed   = math.ceil(total / 10)
    proc     = ops * 10
    avg_talk = None
    if "通話時間" in day_in.columns:
        talks = day_in[day_in["入電結果"] == True]["通話時間"].dropna()
        if not talks.empty:
            avg_talk = int(round(float(talks.mean())))
    return dict(total=total, got=got, missed=missed, rate=rate,
                ops=ops, work_hrs=work_hrs, avg_cph=avg_cph,
                op_per=op_per, needed=needed, proc=proc, avg_talk=avg_talk)


def _prev_month_weekday_hourly_avg(
    cdr: pd.DataFrame, target_date: date, weekday_label: str, in_ext_types: set
) -> dict[int, dict]:
    """前月の同曜日（祝日除外）の時間帯別平均。weekday_labelが「祝」なら祝同士で平均。"""
    import calendar as _cal
    pm_year  = target_date.year if target_date.month > 1 else target_date.year - 1
    pm_month = target_date.month - 1 if target_date.month > 1 else 12
    days_in_pm = _cal.monthrange(pm_year, pm_month)[1]

    hourly_lists: dict[int, list] = {h: [] for h in range(24)}
    for day in range(1, days_in_pm + 1):
        d = date(pm_year, pm_month, day)
        date_csv = str(d).replace("-", "/")
        day_in = cdr[
            (cdr["入電判定"] == True) &
            (cdr["日付"].astype(str).str.contains(date_csv, regex=False)) &
            (cdr["内線種別"].isin(in_ext_types))
        ]
        if day_in.empty:
            continue
        wd = str(day_in["曜日"].iloc[0])
        if weekday_label == "祝":
            if wd != "祝":
                continue
        else:
            if wd != weekday_label or wd == "祝":
                continue
        for h in range(24):
            sub = day_in[day_in["時刻"] == h]
            if len(sub) > 0:
                calls = len(sub)
                ans   = int((sub["入電結果"] == True).sum())
                hourly_lists[h].append({"total": calls, "got": ans, "missed": calls - ans})

    result: dict[int, dict] = {}
    for h in range(24):
        entries = hourly_lists[h]
        if not entries:
            result[h] = {"total": None, "got": None, "missed": None, "rate": None}
        else:
            n  = len(entries)
            t  = sum(e["total"]  for e in entries) / n
            g  = sum(e["got"]    for e in entries) / n
            mi = sum(e["missed"] for e in entries) / n
            result[h] = {
                "total":  round(t,  1),
                "got":    round(g,  1),
                "missed": round(mi, 1),
                "rate":   round(g / t * 100, 1) if t > 0 else None,
            }
    return result


def _hourly_breakdown(cdr: pd.DataFrame, d: date, in_ext_types: set) -> dict[int, dict]:
    date_csv = str(d).replace("-", "/")
    day_in = cdr[
        (cdr["入電判定"] == True) &
        (cdr["日付"].astype(str).str.contains(date_csv)) &
        (cdr["内線種別"].isin(in_ext_types))
    ]
    result: dict[int, dict] = {}
    for h in range(24):
        sub = day_in[day_in["時刻"] == h]
        if len(sub) == 0:
            result[h] = {"total": None, "got": None, "missed": None, "rate": None}
        else:
            calls = len(sub)
            ans   = int((sub["入電結果"] == True).sum())
            result[h] = {
                "total":  calls,
                "got":    ans,
                "missed": calls - ans,
                "rate":   round(ans / calls * 100, 1),
            }
    return result


# ---- analysis --------------------------------------------------------

def analyze(target_date: date,
            cdr: pd.DataFrame, stf: pd.DataFrame,
            ext: pd.DataFrame, ring: pd.DataFrame) -> tuple[str, str]:
    date_str = str(target_date)
    date_csv = date_str.replace("-", "/")

    in_ext_types: set[int] = set(
        pd.to_numeric(ext[ext["Div."] == "IN"]["種別名称一覧 No."], errors="coerce")
        .dropna().astype(int)
    )

    in_df = cdr[
        (cdr["入電判定"] == True) &
        (cdr["日付"].astype(str).str.contains(date_csv)) &
        (cdr["内線種別"].isin(in_ext_types))
    ].copy()

    if in_df.empty:
        msg = "データなし（着信判定=TRUEのレコードが見つかりません）"
        return f"# {date_csv} 分析結果\n\n{msg}\n", f"<html><body><p>{msg}</p></body></html>"

    weekday  = in_df["曜日"].iloc[0]
    miss_df  = in_df[in_df["入電結果"] == False]
    total    = len(in_df)
    got      = int((in_df["入電結果"] == True).sum())
    missed   = total - got
    rate     = round(got / total * 100, 1) if total else 0.0

    ext_type_map: dict[str, str] = {}
    for _, row in in_df[["内線番号", "内線種別"]].dropna().drop_duplicates().iterrows():
        ext_type_map[str(row["内線番号"])] = normalize_type(row["内線種別"])

    capable_map = build_capable_map(ring)

    day_stf      = stf[(stf["Div."] == "IN") & (stf["報告日"].astype(str).str.contains(date_csv))]
    total_work_hrs = float(day_stf["勤務時間"].sum()) if not day_stf.empty else 0.0
    avg_cph      = round(got / total_work_hrs, 1) if total_work_hrs > 0 else None

    avg_talk_secs: int | None = None
    if "通話時間" in in_df.columns:
        talks = in_df[in_df["入電結果"] == True]["通話時間"].dropna()
        if not talks.empty:
            avg_talk_secs = int(round(float(talks.mean())))

    active_ops_by_hour: dict[int, int] = {
        h: len(get_in_ops_by_status(stf, date_csv, h)[0]) for h in range(24)
    }

    # --- 前日・前週の時間帯応答率（hourly比較用） ---
    prev_day_date   = prev_business_day(target_date)
    prev_week_date  = target_date - timedelta(days=7)
    prev_day_hourly  = _hourly_breakdown(cdr, prev_day_date,  in_ext_types)
    prev_week_hourly = _hourly_breakdown(cdr, prev_week_date, in_ext_types)
    prev_month_hourly = _prev_month_weekday_hourly_avg(
        cdr, target_date, weekday, in_ext_types
    )

    # --- 時間帯別 ---
    hourly = []
    for h in range(24):
        sub        = in_df[in_df["時刻"] == h]
        active_cnt = active_ops_by_hour[h]
        over       = active_cnt > 0 and len(sub) > 0 and len(sub) >= active_cnt * 10
        pd_s = prev_day_hourly[h]
        pw_s = prev_week_hourly[h]
        if len(sub) == 0:
            hourly.append({"時間帯": f"{h}時台", "着信": 0, "応答": 0, "未応答": 0,
                           "応答率": None, "判定": "-", "active_ops": active_cnt,
                           "avg_talk_secs": None,
                           "prev_d_rate": pd_s["rate"],   "prev_w_rate":   pw_s["rate"],
                           "prev_d_total": pd_s["total"], "prev_w_total":  pw_s["total"],
                           "prev_d_got":   pd_s["got"],   "prev_w_got":    pw_s["got"],
                           "prev_d_missed":pd_s["missed"],"prev_w_missed": pw_s["missed"],
                           "prev_m_rate":  None, "prev_m_total": None,
                           "prev_m_got":   None, "prev_m_missed": None,
                           "overloaded": over})
            continue
        calls = len(sub)
        ans   = int((sub["入電結果"] == True).sum())
        mis   = calls - ans
        r     = round(ans / calls * 100, 1)
        judge = "✅" if r >= 95 else ("⚠️" if r >= 85 else "🔴")
        if over:
            judge += "⚡"
        avg_talk_h: int | None = None
        if "通話時間" in sub.columns:
            ans_sub = sub[sub["入電結果"] == True]["通話時間"].dropna()
            if not ans_sub.empty:
                avg_talk_h = int(round(float(ans_sub.mean())))
        pm_h = prev_month_hourly[h]
        hourly.append({"時間帯": f"{h}時台", "着信": calls, "応答": ans, "未応答": mis,
                       "応答率": r, "判定": judge, "active_ops": active_cnt,
                       "avg_talk_secs": avg_talk_h,
                       "prev_d_rate": pd_s["rate"],   "prev_w_rate":   pw_s["rate"],
                       "prev_d_total": pd_s["total"], "prev_w_total":  pw_s["total"],
                       "prev_d_got":   pd_s["got"],   "prev_w_got":    pw_s["got"],
                       "prev_d_missed":pd_s["missed"],"prev_w_missed": pw_s["missed"],
                       "prev_m_rate":  pm_h["rate"],  "prev_m_total":  pm_h["total"],
                       "prev_m_got":   pm_h["got"],   "prev_m_missed": pm_h["missed"],
                       "overloaded": over})

    # --- 未応答分類 ---
    miss_classified: list[dict] = []
    for _, row in miss_df.iterrows():
        h               = int(row["時刻"])
        ext_num         = str(row["内線番号"])
        ext_type        = ext_type_map.get(ext_num)
        active, on_break = get_in_ops_by_status(stf, date_csv, h)
        abandon_zoiper  = len(to_zoiper_list(row.get("放棄Zoiper"))) > 0
        label = classify_miss(active, on_break, ext_type, capable_map, abandon_zoiper)
        miss_classified.append({"時刻": f"{h}時台", "内線番号": ext_num,
                                 "案件名": row.get("案件名", ""), "要因": label})

    miss_counts = Counter(r["要因"] for r in miss_classified)

    # --- 内線別 ---
    by_ext = in_df.groupby("内線番号").apply(lambda x: pd.Series({
        "案件名":  x["案件名"].iloc[0] if "案件名" in x.columns else "",
        "内線種別": normalize_type(x["内線種別"].iloc[0]) if "内線種別" in x.columns else "",
        "着信":    len(x),
        "応答":    int((x["入電結果"] == True).sum()),
        "未応答":  int((x["入電結果"] == False).sum()),
        "応答率":  round((x["入電結果"] == True).sum() / len(x) * 100, 1),
        "平均通話時間秒": (
            int(round(float(x[x["入電結果"] == True]["通話時間"].dropna().mean())))
            if "通話時間" in x.columns
               and not x[x["入電結果"] == True]["通話時間"].dropna().empty
            else None
        ),
    })).reset_index()

    all3 = by_ext.copy()
    ext_by_volume = by_ext.sort_values(["着信", "応答率"], ascending=[False, False]).head(5)
    ext_best5  = all3.sort_values(["応答率", "着信"], ascending=[False, False]).head(5)
    ext_worst5 = all3.sort_values(["応答率", "着信"], ascending=[True,  False]).head(5)
    ext_special = (
        all3[
            all3["案件名"].apply(lambda n: get_ext_type_label(str(n))).isin(["A", "B"])
            & (all3["応答率"] < 90)
        ]
        .sort_values(["応答率", "着信"], ascending=[True, False])
    )

    # --- 案件別日次データ（時間帯別含む） ---
    ext_daily_list: list[dict] = []
    for _, row in all3.sort_values("着信", ascending=False).iterrows():
        ext_key = str(row["内線番号"])
        ext_sub = in_df[in_df["内線番号"].astype(str) == ext_key]
        h_total = [0] * 24
        h_got   = [0] * 24
        for h in range(24):
            hour_sub = ext_sub[ext_sub["時刻"] == h]
            if len(hour_sub) > 0:
                h_total[h] = len(hour_sub)
                h_got[h]   = int((hour_sub["入電結果"] == True).sum())
        _ats = row.get("平均通話時間秒")
        _ats_int = int(_ats) if pd.notna(_ats) and _ats is not None else None
        ext_daily_list.append({
            "ext":          ext_key,
            "name":         str(row["案件名"]),
            "type":         str(row["内線種別"]),
            "total":        int(row["着信"]),
            "got":          int(row["応答"]),
            "missed":       int(row["未応答"]),
            "rate":         float(row["応答率"]),
            "kpi":          kpi_label(float(row["応答率"])),
            "h":            h_total,
            "g":            h_got,
            "avg_talk_secs": _ats_int,
            "talk_sum":     (_ats_int * int(row["応答"])) if _ats_int is not None else 0,
        })

    # --- OP別 ---
    op_stats: dict[str, dict] = {}
    for _, row in day_stf.iterrows():
        z = to_zoiper_str(row.get("Zoiper"))
        if z is None:
            continue
        try:
            wh = float(row.get("勤務時間", 0) or 0)
        except Exception:
            wh = 0.0
        op_stats[z] = {"OP名": str(row.get("OP名", "")), "稼働時間": wh,
                       "応答": 0, "放棄着信": 0, "通話合計秒": 0, "通話件数": 0}

    for _, row in in_df.iterrows():
        z_got   = to_zoiper_str(row.get("取得Zoiper"))
        z_drops = to_zoiper_list(row.get("放棄Zoiper"))
        if z_got and z_got in op_stats and row.get("入電結果") == True:
            op_stats[z_got]["応答"] += 1
            if "通話時間" in row.index and pd.notna(row["通話時間"]):
                op_stats[z_got]["通話合計秒"] += int(row["通話時間"])
                op_stats[z_got]["通話件数"]   += 1
        for z_drop in z_drops:
            if z_drop in op_stats:
                op_stats[z_drop]["放棄着信"] += 1

    op_rows = []
    for z, s in op_stats.items():
        ans_cnt = s["応答"]
        mis_cnt = s["放棄着信"]
        total_z = ans_cnt + mis_cnt
        r_z     = round(ans_cnt / total_z * 100, 1) if total_z > 0 else None
        wh      = s["稼働時間"]
        cph     = round(ans_cnt / wh, 1) if wh > 0 and ans_cnt > 0 else None
        avg_t_op = (int(round(s["通話合計秒"] / s["通話件数"]))
                    if s["通話件数"] > 0 else None)
        op_rows.append({"Zoiper": z, "OP名": s["OP名"], "着信": total_z,
                        "応答": ans_cnt, "未応答": mis_cnt, "応答率": r_z,
                        "稼働時間": wh, "CPH": cph, "平均通話時間秒": avg_t_op})

    op_df = pd.DataFrame(op_rows)
    if not op_df.empty:
        cph_df  = op_df[op_df["CPH"].notna()]
        rate_df = op_df[op_df["応答率"].notna()]
        op_cph_best5   = cph_df.sort_values( ["CPH",   "着信"], ascending=[False, False]).head(5)
        op_cph_worst5  = cph_df.sort_values( ["CPH",   "着信"], ascending=[True,  False]).head(5)
        op_rate_best5  = rate_df.sort_values(["応答率", "着信"], ascending=[False, False]).head(5)
        op_rate_worst5 = rate_df.sort_values(["応答率", "着信"], ascending=[True,  False]).head(5)
    else:
        op_cph_best5 = op_cph_worst5 = op_rate_best5 = op_rate_worst5 = pd.DataFrame()

    # --- OP別時間帯データ（Zoiperクロス集計用） ---
    _ans_df = in_df[in_df["入電結果"] == True].copy()
    _ans_df["_z_str"] = _ans_df["取得Zoiper"].apply(to_zoiper_str)
    _miss_df = in_df[in_df["入電結果"] == False].copy()

    # 種別別未応答数を放棄Zoiperごとに集計: {z: {type: count}}
    _miss_type_by_z: dict[str, dict[str, int]] = {}
    for _, _mrow in _miss_df.iterrows():
        _zt = normalize_type(_mrow.get("内線種別"))
        if _zt is None:
            continue
        _zt = str(_zt)
        for _zd in to_zoiper_list(_mrow.get("放棄Zoiper")):
            if _zd not in _miss_type_by_z:
                _miss_type_by_z[_zd] = {}
            _miss_type_by_z[_zd][_zt] = _miss_type_by_z[_zd].get(_zt, 0) + 1

    op_daily_list: list[dict] = []
    for row_data in op_rows:
        z = row_data["Zoiper"]
        z_ans = _ans_df[_ans_df["_z_str"] == z]
        h_got_op = [int((z_ans["時刻"] == h).sum()) for h in range(24)]
        # 種別別応答数
        type_got_op: dict[str, int] = {}
        if not z_ans.empty and "内線種別" in z_ans.columns:
            for t_val, cnt in z_ans["内線種別"].apply(normalize_type).value_counts().items():
                if t_val is not None:
                    type_got_op[str(t_val)] = int(cnt)
        # 種別別着信数（応答数 + 未応答数）
        type_missed_op = _miss_type_by_z.get(z, {})
        type_total_op: dict[str, int] = {}
        all_types = set(type_got_op) | set(type_missed_op)
        for _t in all_types:
            type_total_op[_t] = type_got_op.get(_t, 0) + type_missed_op.get(_t, 0)
        s = op_stats.get(z, {})
        talk_sum_op = int(s.get("通話合計秒", 0) or 0)
        talk_cnt_op = int(s.get("通話件数", 0) or 0)
        avg_talk_op = int(round(talk_sum_op / talk_cnt_op)) if talk_cnt_op > 0 else None
        op_daily_list.append({
            "zoiper":         z,
            "name":           row_data["OP名"],
            "total":          row_data["着信"],
            "got":            row_data["応答"],
            "missed":         row_data["未応答"],
            "rate":           float(row_data["応答率"]) if row_data["応答率"] is not None else None,
            "work_hrs":       float(row_data["稼働時間"]),
            "cph":            float(row_data["CPH"]) if row_data["CPH"] is not None else None,
            "avg_talk_secs":  avg_talk_op,
            "talk_sum":       talk_sum_op,
            "h_got":          h_got_op,
            "type_got":       type_got_op,
            "type_total":     type_total_op,
        })

    # --- 前週比・前月同曜日平均比 ---
    prev_day_s  = _day_stats(cdr, stf, prev_day_date,  in_ext_types)
    prev_week_s = _day_stats(cdr, stf, prev_week_date, in_ext_types)

    # 前週が祝日かどうか
    pw_day_in = cdr[
        (cdr["入電判定"] == True) &
        (cdr["日付"].astype(str).str.contains(
            str(prev_week_date).replace("-", "/"), regex=False)) &
        (cdr["内線種別"].isin(in_ext_types))
    ]
    prev_week_is_holiday = (
        not pw_day_in.empty and str(pw_day_in["曜日"].iloc[0]) == "祝"
    )

    # 前月同曜日平均（祝日除外）
    prev_month_avg_s = _prev_month_weekday_avg(
        cdr, stf, target_date, weekday, in_ext_types
    )

    # --- 通話時間ランキング ---
    if "通話時間" in in_df.columns and not by_ext.empty:
        talk_ranking = (
            by_ext[by_ext["平均通話時間秒"].notna()]
            .sort_values("平均通話時間秒", ascending=False)
            .head(5)
        )
    else:
        talk_ranking = pd.DataFrame()

    hourly_calls: dict[int, list[dict]] = {}
    for h in range(24):
        sub = in_df[in_df["時刻"] == h]
        calls_list = []
        for _, row in sub.iterrows():
            nt = normalize_type(row.get("内線種別"))
            calls_list.append({
                "入電日時":   str(row.get("入電日時", "")),
                "内線番号":   str(row.get("内線番号", "")),
                "案件名":     str(row.get("案件名", "")),
                "内線種別":   nt if nt else "-",
                "結果":       "応答" if row.get("入電結果") is True else "未応答",
                "取得Zoiper": to_zoiper_str(row.get("取得Zoiper")) or "-",
                "放棄Zoiper": ", ".join(to_zoiper_list(row.get("放棄Zoiper"))) or "-",
                "呼出時間":   str(row.get("呼出時間", "")),
                "通話時間":   str(row.get("通話時間", "")),
            })
        calls_list.sort(key=lambda c: c["入電日時"])
        hourly_calls[h] = calls_list

    data = dict(
        date_csv=date_csv, weekday=weekday,
        total=total, got=got, missed=missed,
        rate=rate,
        total_work_hrs=total_work_hrs, avg_cph=avg_cph,
        avg_talk_secs=avg_talk_secs,
        prev_day_s=prev_day_s, prev_week_s=prev_week_s,
        prev_month_avg_s=prev_month_avg_s,
        prev_week_is_holiday=prev_week_is_holiday,
        hourly=hourly,
        miss_counts=miss_counts,
        ext_best5=ext_best5, ext_worst5=ext_worst5, ext_by_volume=ext_by_volume,
        ext_special=ext_special, ext_daily=ext_daily_list,
        op_cph_best5=op_cph_best5, op_cph_worst5=op_cph_worst5,
        op_rate_best5=op_rate_best5, op_rate_worst5=op_rate_worst5,
        op_daily=op_daily_list,
        hourly_calls=hourly_calls,
        talk_ranking=talk_ranking,
        miss_classified=miss_classified,
    )
    return build_html(data)


# ---- report helpers --------------------------------------------------

# ---- HTML builder ----------------------------------------------------

_CSS = """
<style>
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans JP','Hiragino Sans','Yu Gothic UI',sans-serif;margin:2em;color:#333}
h1{color:#222}
h2{color:#444;border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:1.8em}
h3,h4{color:#555}
table{border-collapse:collapse;margin:0.8em 0}
th,td{border:1px solid #ccc;padding:5px 10px;text-align:center}
th{background:#f0f0f0;text-align:center}
.green{background:#d4edda}
.yellow{background:#fff3cd}
.red{background:#f8d7da}
.alert-high{color:#721c24;font-weight:bold}
.alert-mid{color:#856404}
hr{border:1px solid #ddd;margin:1.5em 0}
.side-by-side{display:flex;gap:1.5em;flex-wrap:wrap;align-items:flex-start}
.side-by-side>div{flex:1;min-width:0}
.hourly-wrap{overflow-x:auto;margin:0.8em 0}
.hourly-wrap table{min-width:max-content;white-space:nowrap}
.hourly-wrap th{background:#f0f0f0;text-align:center}
.hourly-wrap th:first-child{text-align:left;position:sticky;left:0;z-index:2;background:#f0f0f0}
.comp-row td,.comp-row th{font-size:0.82em;color:#555;background:#f8f8f8}
.comp-row th{padding-left:1.5em}
.comp-row th:first-child{background:#f8f8f8}
.search-bar{background:#f8f8f8;border:1px solid #ddd;border-radius:6px;padding:8px 14px;margin-bottom:1.5em;display:flex;align-items:center;gap:12px}
.search-bar input{padding:6px 10px;border:1px solid #aaa;border-radius:4px;font-size:0.95em;width:280px}
.search-bar label{font-weight:bold;color:#444;white-space:nowrap}
.search-count{font-size:0.9em;color:#666}
.hour-hdr{cursor:pointer;user-select:none;background:#f0f4ff !important}
.hour-hdr:hover{background:#dde8ff !important}
.hour-hdr::after{content:" ▼";font-size:0.75em;color:#666}
.hour-hdr[data-open="1"]{background:#4a90d9 !important;color:#fff !important}
.hour-hdr[data-open="1"]::after{content:" ▲";font-size:0.75em;color:#fff}
.detail-row>td{padding:0;background:#fafafa}
.ab-row{cursor:pointer;user-select:none}
.ab-row:hover{background:#dde8ff !important}
.ab-row td:first-child::before{content:"▶ ";font-size:0.75em;color:#666}
.ab-row[data-open="1"]{background:#4a90d9 !important;color:#fff !important}
.ab-row[data-open="1"] td:first-child::before{content:"▼ ";color:#fff}
.ab-row[data-open="1"] .rate-ok,.ab-row[data-open="1"] .rate-warn,.ab-row[data-open="1"] .rate-ng{background:inherit !important;color:#fff !important}
.inner-table{width:100%;border-collapse:collapse;font-size:0.88em}
.inner-table th,.inner-table td{border:1px solid #ddd;padding:3px 8px}
.inner-table th{background:#e0e0e0;text-align:center}
.inner-table td{text-align:center}
.result-ok{color:#155724;font-weight:bold}
.result-ng{color:#721c24;font-weight:bold}
.kpi-badge{display:inline-block;padding:1px 7px;border-radius:3px;font-weight:bold;font-size:0.85em;letter-spacing:0.03em}
.kpi-s{background:#0d6efd;color:#fff}
.kpi-a{background:#198754;color:#fff}
.kpi-b{background:#20c997;color:#fff}
.kpi-c{background:#ffc107;color:#000}
.kpi-d{background:#dc3545;color:#fff}
</style>
"""


def _compute_factors(d: dict) -> dict[str, list[str]]:
    factors: dict[str, list[str]] = {"需要": [], "供給": [], "生産性": []}
    pm    = d.get("prev_month_avg_s")
    pw    = d["prev_week_s"]
    total = d["total"]

    # --- 需要要因 ---
    if pm and (pm.get("total") or 0) > 0:
        ratio = (total - pm["total"]) / pm["total"]
        if abs(ratio) >= 0.10:
            factors["需要"].append(
                f"着信数が前月同曜日平均比 {round(ratio * 100, 1):+.1f}%（{'増加' if ratio > 0 else '減少'}）")
    if pw and (pw.get("total") or 0) > 0:
        ratio = (total - pw["total"]) / pw["total"]
        if abs(ratio) >= 0.10:
            factors["需要"].append(
                f"着信数が前週同曜日比 {round(ratio * 100, 1):+.1f}%（{'増加' if ratio > 0 else '減少'}）")
    if total > 0:
        for h in range(24):
            cnt = d["hourly"][h]["着信"]
            if cnt > 0 and cnt / total >= 0.20:
                pct = round(cnt / total * 100, 1)
                factors["需要"].append(f"{h}時台に着信集中（{cnt}件・全体の{pct}%）")

    # --- 供給要因 ---
    shortage = []
    for h in range(24):
        hr    = d["hourly"][h]
        calls = hr["着信"]
        ops   = hr["active_ops"]
        if calls > 0 and ops < math.ceil(calls / 10):
            needed = math.ceil(calls / 10)
            shortage.append(f"{h}時台（必要{needed}名・実績{ops}名）")
    if shortage:
        suffix = "…ほか" if len(shortage) > 5 else ""
        factors["供給"].append(
            f"人員不足時間帯：{' / '.join(shortage[:5])}{suffix}")
    if total > 0:
        active_hours = [h for h in range(24) if d["hourly"][h]["着信"] > 0]
        processable  = sum(d["hourly"][h]["active_ops"] * 10 for h in active_hours)
        if processable > 0:
            dev = (total - processable) / total
            if dev > 0.10:
                factors["供給"].append(
                    f"処理可能数超過（着信{total}件 / 処理可能{processable}件・乖離率{round(dev * 100, 1)}%）")
            elif dev < -0.10:
                factors["供給"].append(
                    f"人員余剰の可能性（着信{total}件 / 処理可能{processable}件）")

    # --- 生産性要因 ---
    cph = d.get("avg_cph")
    if cph is not None:
        if cph < 8:
            factors["生産性"].append(f"平均CPHが低い（{cph}件/h・目安8以上）")
        elif cph >= 12:
            factors["生産性"].append(f"平均CPHが高い（{cph}件/h）")
        if pm and (pm.get("avg_cph") or 0) > 0:
            ratio = (cph - pm["avg_cph"]) / pm["avg_cph"]
            if abs(ratio) >= 0.20:
                factors["生産性"].append(
                    f"CPHが前月同曜日平均比 {round(ratio * 100, 1):+.1f}%（{'上昇' if ratio > 0 else '低下'}）")
    secs = d.get("avg_talk_secs")
    if secs is not None:
        if secs > 300:
            factors["生産性"].append(f"平均通話時間が長い（{fmt_seconds(secs)}）")
        if pm and (pm.get("avg_talk") or 0) > 0:
            ratio = (secs - pm["avg_talk"]) / pm["avg_talk"]
            if abs(ratio) >= 0.20:
                factors["生産性"].append(
                    f"平均通話時間が前月同曜日平均比 {round(ratio * 100, 1):+.1f}%（{'増加' if ratio > 0 else '減少'}）")

    return factors


def build_factors_html(d: dict) -> str:
    factors = _compute_factors(d)
    if all(not v for v in factors.values()):
        return '<p style="color:#888;font-size:0.9em">自動判定できる要因はありません</p>\n'
    html = '<table style="width:auto">\n'
    for label in ["需要", "供給", "生産性"]:
        items = factors[label]
        n = len(items)
        if n == 0:
            html += (f'<tr><td style="text-align:left;font-weight:bold;white-space:nowrap;'
                     f'padding-right:1em;vertical-align:top">{label}要因</td>'
                     f'<td style="text-align:left;color:#aaa">確認できません</td></tr>\n')
        else:
            for i, item in enumerate(items):
                if i == 0:
                    rs = f' rowspan="{n}"' if n > 1 else ""
                    html += (f'<tr><td style="text-align:left;font-weight:bold;white-space:nowrap;'
                             f'padding-right:1em;vertical-align:top"{rs}>{label}要因</td>'
                             f'<td style="text-align:left">{item}</td></tr>\n')
                else:
                    html += f'<tr><td style="text-align:left">{item}</td></tr>\n'
    html += '</table>\n'
    html += ('<p style="color:#888;font-size:0.82em;margin-top:0.4em">'
             '※ ログデータによる自動判定。品質・システム要因は手動確認が必要です。</p>\n')
    return html


def _rate_td(r) -> str:
    if r is None:
        return "<td>-</td>"
    cls = rate_class(r)
    return f'<td class="{cls}">{r}%</td>'


def _html_table(headers: list[str], rows_html: str) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    return f"<table><tr>{ths}</tr>{rows_html}</table>"


_TOOLTIP_WEEK_HOLIDAY  = "前週が祝日のため参考値外"
_TOOLTIP_NO_PREV_MONTH = "前月に同曜日のデータなし"


def build_html(d: dict) -> str:
    pm  = d.get("prev_month_avg_s")
    pw  = d["prev_week_s"]
    pw_holiday = d.get("prev_week_is_holiday", False)

    def _tooltip_cell(text: str, tip: str) -> str:
        return (f'<span title="{tip}" style="cursor:help;color:#aaa;'
                f'border-bottom:1px dashed #aaa">{text}</span>')

    def dcell_plain(cur, pm_key: str, unit: str = "件", is_rate: bool = False,
                    use_float: bool = False) -> tuple[str, str, bool, bool]:
        """Plain-text comparison. Returns (month, week, month_missing, week_holiday).
        Used for both the HTML table (wrapped by dcell) and the JSON for notify.py."""
        # 前月同曜日平均比
        pm_val = pm[pm_key] if (pm and pm_key in pm and pm[pm_key] is not None) else None
        month_missing = pm_val is None
        month_s = "－" if month_missing else _diff_str(cur, pm_val, unit, is_rate, use_float)

        # 前週同曜日比（前週が祝日なら参考値外）
        if pw_holiday:
            week_s = "－"
        else:
            pw_val = pw[pm_key] if (pw and pm_key in pw) else None
            week_s = _diff_str(cur, pw_val, unit, is_rate, use_float)

        return (month_s, week_s, month_missing, pw_holiday)

    def dcell(cur, pm_key: str, unit: str = "件", is_rate: bool = False,
              use_float: bool = False) -> tuple[str, str]:
        month_s, week_s, month_missing, week_holiday = dcell_plain(
            cur, pm_key, unit, is_rate, use_float)
        if month_missing:
            month_s = _tooltip_cell(month_s, _TOOLTIP_NO_PREV_MONTH)
        if week_holiday:
            week_s = _tooltip_cell(week_s, _TOOLTIP_WEEK_HOLIDAY)
        return (month_s, week_s)

    def srow(label: str, cur_disp: str, cur_raw, key: str, unit: str = "件",
             is_rate: bool = False, use_float: bool = False) -> str:
        month_s, week_s = dcell(cur_raw, key, unit, is_rate, use_float)
        return (f'<tr><td style="text-align:left">{label}</td><td><strong>{cur_disp}</strong></td>'
                f"<td>{month_s}</td><td>{week_s}</td></tr>")

    cph_disp  = str(d["avg_cph"]) if d["avg_cph"] is not None else "-"
    talk_disp = fmt_seconds(d["avg_talk_secs"])
    cph_raw   = d["avg_cph"] if d["avg_cph"] is not None else 0
    talk_raw  = d["avg_talk_secs"] if d["avg_talk_secs"] is not None else 0

    summary_rows = (
        srow("着信数",       f"{d['total']}件",             d["total"],          "total",    "件")
      + srow("応答数",       f"{d['got']}件",               d["got"],            "got",      "件")
      + srow("未応答数",     f"{d['missed']}件",            d["missed"],         "missed",   "件")
      + srow("応答率",       f"{d['rate']}%",               d["rate"],           "rate",     "%",  True)
      + srow("合計稼働時間", f"{d['total_work_hrs']:.1f}h", d["total_work_hrs"], "work_hrs", "h",  False, True)
      + srow("平均CPH",      cph_disp,                      cph_raw,             "avg_cph",  "",   False, True)
      + srow("平均通話時間", talk_disp,                     talk_raw,            "avg_talk", "秒")
    )

    miss_rows = ""
    for label, cnt in sorted(d["miss_counts"].items()):
        pct = round(cnt / d["missed"] * 100, 1) if d["missed"] else 0
        miss_rows += f'<tr><td style="text-align:left">{label}</td><td>{cnt}件</td><td>{pct}%</td></tr>'

    # --- Transposed hourly table (all 24 hours, integrated collapsible) ---
    all_hours = list(range(24))

    def _tds(vals: list) -> str:
        return "".join(f"<td>{v}</td>" for v in vals)

    def _hrow(label: str, cells_html: str, cls: str = "") -> str:
        cl = f' class="{cls}"' if cls else ""
        return (f'<tr{cl}><th style="text-align:left;white-space:nowrap;'
                f'min-width:7em">{label}</th>{cells_html}</tr>')

    def _talk_td_h(h: int) -> str:
        v = d["hourly"][h].get("avg_talk_secs")
        return f"<td>{fmt_seconds(v)}</td>"

    def _overage_td(h: int) -> str:
        calls  = d["hourly"][h]["着信"]
        ops    = d["hourly"][h]["active_ops"]
        needed = math.ceil(calls / 10) if calls > 0 else 0
        diff   = ops - needed
        if diff > 0:
            return f"<td>+{diff}</td>"
        if diff < 0:
            return f"<td>{diff}</td>"
        return "<td>±0</td>"

    h_headers = "<th>合計</th>"
    for h in all_hours:
        hr = d["hourly"][h]
        if hr["着信"] > 0:
            h_headers += (f'<th class="hour-hdr" id="arrow-{h}" '
                          f'onclick="toggleDetail({h})" data-open="">{h}時台</th>')
        else:
            h_headers += f"<th>{h}時台</th>"

    detail_rows = ""
    for h in all_hours:
        calls = d["hourly_calls"].get(h, [])
        if not calls:
            continue
        inner = ""
        for c in calls:
            res_cls = "result-ok" if c["結果"] == "応答" else "result-ng"
            inner += (f'<tr><td style="text-align:left">{c["入電日時"]}</td>'
                      f'<td>{c["内線種別"]}</td>'
                      f'<td>{c["内線番号"]}</td>'
                      f'<td style="text-align:left">{c["案件名"]}</td>'
                      f'<td class="{res_cls}">{c["結果"]}</td>'
                      f'<td>{c["取得Zoiper"]}</td>'
                      f'<td style="text-align:left">{c["放棄Zoiper"]}</td>'
                      f'<td>{c["呼出時間"]}</td>'
                      f'<td>{c["通話時間"]}</td></tr>')
        detail_rows += (
            f'<tr class="detail-row" id="detail-{h}" style="display:none">'
            f'<td colspan="26">'
            f'<table class="inner-table" style="width:auto">'
            f'<colgroup><col style="width:10em"><col style="width:3em"><col style="width:4.5em">'
            f'<col style="min-width:10em"><col style="width:4em"><col style="width:4em">'
            f'<col style="width:14em"><col style="width:4em"><col style="width:4em"></colgroup>'
            f'<tr><th>着信日時</th><th>種別</th><th>内線</th><th>案件名</th><th>結果</th>'
            f'<th>取得Zoiper</th><th>放棄Zoiper</th><th>呼出時間</th><th>通話時間</th></tr>'
            f'{inner}</table></td></tr>\n'
        )

    ps = d.get("prev_day_s")  # hourly table still uses prev_day for row-level detail

    def _h_tot_diff(cur, pm_key: str, pw_key: str, unit: str = "件", is_rate: bool = False) -> tuple[str, str]:
        pm_v = pm[pm_key] if (pm and pm_key in pm and pm[pm_key] is not None) else None
        pw_v = pw[pw_key] if (pw and pw_key in pw) else None
        m_cell = _diff_str(cur, pm_v, unit, is_rate) if pm_v is not None else (
            f'<span title="{_TOOLTIP_NO_PREV_MONTH}" style="cursor:help;color:#aaa;border-bottom:1px dashed #aaa">－</span>'
        )
        if pw_holiday:
            w_cell = f'<span title="{_TOOLTIP_WEEK_HOLIDAY}" style="cursor:help;color:#aaa;border-bottom:1px dashed #aaa">－</span>'
        else:
            w_cell = _diff_str(cur, pw_v, unit, is_rate)
        return (f"<td>{m_cell}</td>", f"<td>{w_cell}</td>")

    tot_m_total,  tot_w_total  = _h_tot_diff(d["total"],  "total",    "total",    "件")
    tot_m_got,    tot_w_got    = _h_tot_diff(d["got"],    "got",      "got",      "件")
    tot_m_missed, tot_w_missed = _h_tot_diff(d["missed"], "missed",   "missed",   "件")
    tot_m_rate,   tot_w_rate   = _h_tot_diff(d["rate"],   "rate",     "rate",     "%", True)

    if any(r["着信"] > 0 for r in d["hourly"]):
        hourly_table_html = (
            '<div class="hourly-wrap"><table>'
            f'<tr><th></th>{h_headers}</tr>'
            + _hrow("着信数",      f"<td>{d['total']}</td>"   + _tds([d["hourly"][h]["着信"]   for h in all_hours]))
            + _hrow("　前月平均比", tot_m_total  + _tds([_diff_str(d["hourly"][h]["着信"],   d["hourly"][h]["prev_m_total"],  "件") for h in all_hours]), "comp-row")
            + _hrow("　前週比",    tot_w_total  + _tds([_diff_str(d["hourly"][h]["着信"],   d["hourly"][h]["prev_w_total"],  "件") for h in all_hours]), "comp-row")
            + _hrow("応答数",      f"<td>{d['got']}</td>"     + _tds([d["hourly"][h]["応答"]   for h in all_hours]))
            + _hrow("　前月平均比", tot_m_got    + _tds([_diff_str(d["hourly"][h]["応答"],   d["hourly"][h]["prev_m_got"],    "件") for h in all_hours]), "comp-row")
            + _hrow("　前週比",    tot_w_got    + _tds([_diff_str(d["hourly"][h]["応答"],   d["hourly"][h]["prev_w_got"],    "件") for h in all_hours]), "comp-row")
            + _hrow("未応答数",    f"<td>{d['missed']}</td>"  + _tds([d["hourly"][h]["未応答"]  for h in all_hours]))
            + _hrow("　前月平均比", tot_m_missed + _tds([_diff_str(d["hourly"][h]["未応答"], d["hourly"][h]["prev_m_missed"], "件") for h in all_hours]), "comp-row")
            + _hrow("　前週比",    tot_w_missed + _tds([_diff_str(d["hourly"][h]["未応答"], d["hourly"][h]["prev_w_missed"], "件") for h in all_hours]), "comp-row")
            + _hrow("応答率",      _rate_td(d["rate"]) + "".join(_rate_td(d["hourly"][h]["応答率"]) for h in all_hours))
            + _hrow("　前月平均比", tot_m_rate   + _tds([_diff_str(d["hourly"][h]["応答率"], d["hourly"][h]["prev_m_rate"],   "%", is_rate=True) for h in all_hours]), "comp-row")
            + _hrow("　前週比",    tot_w_rate   + _tds([_diff_str(d["hourly"][h]["応答率"], d["hourly"][h]["prev_w_rate"],   "%", is_rate=True) for h in all_hours]), "comp-row")
            + _hrow("人員数",    "<td>-</td>" + _tds([d["hourly"][h]["active_ops"] for h in all_hours]))
            + _hrow("過不足",    "<td>-</td>" + "".join(_overage_td(h) for h in all_hours))
            + _hrow("平均通話時間", f"<td>{fmt_seconds(d['avg_talk_secs'])}</td>" + "".join(_talk_td_h(h) for h in all_hours))
            + detail_rows
            + "</table></div>"
        )
    else:
        hourly_table_html = "<p>データなし</p>"

    def _ext_rows_html(df: pd.DataFrame) -> str:
        rows = ""
        for i, (_, row) in enumerate(df.iterrows(), 1):
            avg_t = fmt_seconds(row.get("平均通話時間秒"))
            rows += (f"<tr><td>{i}</td><td>{row['内線種別']}</td><td>{row['内線番号']}</td>"
                     f'<td style="text-align:left">{row["案件名"]}</td>'
                     f"<td>{row['着信']}</td><td>{row['応答']}</td>"
                     f"<td>{row['未応答']}</td>{_rate_td(row['応答率'])}"
                     f"{kpi_badge_html(row['応答率'])}<td>{avg_t}</td></tr>")
        return rows

    ext_headers = ["順位", "種別", "内線", "案件名", "着信", "応答", "未応答", "応答率", "KPI", "平均通話時間"]

    def _cph_td(cph) -> str:
        if cph is None:
            return "<td>-</td>"
        if cph >= 10:
            return f'<td class="green">{cph}</td>'
        if cph >= 8:
            return f'<td class="yellow">{cph}</td>'
        return f'<td class="red">{cph}</td>'

    def _op_cph_rows_html(df: pd.DataFrame) -> str:
        rows = ""
        for i, (_, row) in enumerate(df.iterrows(), 1):
            avg_t = fmt_seconds(row.get("平均通話時間秒"))
            rows += (f"<tr><td>{i}</td><td>{row['Zoiper']}</td>"
                     f'<td style="text-align:left">{row["OP名"]}</td>'
                     f"<td>{row['着信']}</td><td>{row['応答']}</td><td>{row['未応答']}</td>"
                     f"{_rate_td(row['応答率'])}"
                     f"<td>{row['稼働時間']}h</td>{_cph_td(row['CPH'])}<td>{avg_t}</td></tr>")
        return rows

    def _op_rate_rows_html(df: pd.DataFrame) -> str:
        rows = ""
        for i, (_, row) in enumerate(df.iterrows(), 1):
            avg_t = fmt_seconds(row.get("平均通話時間秒"))
            rows += (f"<tr><td>{i}</td><td>{row['Zoiper']}</td>"
                     f'<td style="text-align:left">{row["OP名"]}</td>'
                     f"<td>{row['着信']}</td><td>{row['応答']}</td><td>{row['未応答']}</td>"
                     f"{_rate_td(row['応答率'])}"
                     f"<td>{row['稼働時間']}h</td>{_cph_td(row['CPH'])}<td>{avg_t}</td></tr>")
        return rows

    op_cph_headers  = ["順位", "Zoiper", "OP名", "着信", "応答", "未応答", "応答率", "稼働時間", "CPH", "平均通話時間"]
    op_rate_headers = ["順位", "Zoiper", "OP名", "着信", "応答", "未応答", "応答率", "稼働時間", "CPH", "平均通話時間"]

    def _talk_rows_html(df: pd.DataFrame) -> str:
        rows = ""
        for i, (_, row) in enumerate(df.iterrows(), 1):
            rows += (f"<tr><td>{i}</td><td>{row['内線種別']}</td><td>{row['内線番号']}</td>"
                     f'<td style="text-align:left">{row["案件名"]}</td>'
                     f"<td>{row['着信']}</td><td>{row['応答']}</td><td>{row['未応答']}</td>"
                     f"{_rate_td(row['応答率'])}"
                     f"<td>{fmt_seconds(row['平均通話時間秒'])}</td></tr>")
        return rows

    talk_headers = ["順位", "種別", "内線", "案件名", "着信", "応答", "未応答", "応答率", "平均通話時間"]
    talk_section_html = (
        _html_table(talk_headers, _talk_rows_html(d["talk_ranking"]))
        if not d["talk_ranking"].empty
        else "<p style='color:#888;font-size:0.9em'>通話時間データなし</p>"
    )

    # --- 特定案件ランキング ([A]/[B], 応答率90%未満) ---
    _miss_cls_list = d.get("miss_classified", [])
    _miss_by_ph: dict = {}
    for _m in _miss_cls_list:
        try:
            _mh = int(_m["時刻"].replace("時台", ""))
        except Exception:
            continue
        _mk = (_m["案件名"], _mh)
        if _mk not in _miss_by_ph:
            _miss_by_ph[_mk] = []
        _miss_by_ph[_mk].append(_m["要因"])

    _ab_by_proj: dict = {}
    for _h in range(24):
        for _c in d["hourly_calls"].get(_h, []):
            if get_ext_type_label(_c["案件名"]) in ("A", "B"):
                _pn = _c["案件名"]
                if _pn not in _ab_by_proj:
                    _ab_by_proj[_pn] = []
                _ab_by_proj[_pn].append((_h, _c))

    _ab_stats = []
    for _pn, _items in _ab_by_proj.items():
        _pt = len(_items)
        _pm = sum(1 for _, _c in _items if _c["結果"] == "未応答")
        _pg = _pt - _pm
        _pr = round(_pg / _pt * 100, 1) if _pt > 0 else 100.0
        _talks = []
        for _, _c in _items:
            try:
                _tv = int(_c.get("通話時間", "") or "")
                if _tv > 0:
                    _talks.append(_tv)
            except (ValueError, TypeError):
                pass
        _avg_talk = round(sum(_talks) / len(_talks)) if _talks else None
        _ab_stats.append({"name": _pn, "total": _pt, "got": _pg, "missed": _pm,
                          "rate": _pr, "avg_talk": _avg_talk})

    _ab_stats = [_p for _p in _ab_stats if _p["rate"] < 90]
    _ab_stats.sort(key=lambda x: (x["rate"], -x["total"]))

    _zoiper_name: dict[str, str] = {
        op["zoiper"]: op["name"] for op in d.get("op_daily", [])
    }

    if _ab_stats:
        ab_ranking_html = (
            '<div style="margin:1em 0">'
            '<div style="font-weight:bold;font-size:1.1em;padding:4px 0">'
            '▼ 特定案件ランキング（[A]/[B]・応答率90%未満）</div>\n'
            '<table><tr>'
            '<th style="text-align:left">案件名</th>'
            '<th>着信</th><th>応答</th><th>未応答</th><th>応答率</th><th>KPI</th><th>平均通話時間</th>'
            '</tr>\n'
        )
        for _pi, _p in enumerate(_ab_stats):
            _rid = f'ab-row-{_pi}'
            _did = f'ab-det-{_pi}'
            ab_ranking_html += (
                f'<tr class="ab-row" id="{_rid}" data-open="" title="クリックで詳細を展開"'
                f' onclick="toggleAbRow(\'{_rid}\',\'{_did}\')">'
                f'<td style="text-align:left">{_p["name"]}</td>'
                f'<td>{_p["total"]}</td><td>{_p["got"]}</td><td>{_p["missed"]}</td>'
                f'{_rate_td(_p["rate"])}'
                f'{kpi_badge_html(_p["rate"])}'
                f'<td>{fmt_seconds(_p["avg_talk"])}</td></tr>\n'
            )
            _pn = _p["name"]
            _hg: dict = {}
            for _h, _c in _ab_by_proj[_pn]:
                if _h not in _hg:
                    _hg[_h] = []
                _hg[_h].append(_c)
            _det = (
                '<table><tr><th>時間帯</th><th>稼働OP数</th>'
                '<th style="text-align:left">放棄Zoiper</th>'
                '<th style="text-align:left">取れなかった理由</th></tr>\n'
            )
            for _h in sorted(_hg):
                _clist = _hg[_h]
                _miss_list = [_c for _c in _clist if _c["結果"] == "未応答"]
                if not _miss_list:
                    continue
                _oh = d["hourly"][_h]["active_ops"]
                _rs = _miss_by_ph.get((_pn, _h), [])
                if _rs:
                    _rcnt = Counter(_rs)
                    _rstr = "、".join(
                        f"{_r}({_cnt}件)" for _r, _cnt in sorted(_rcnt.items())
                    )
                else:
                    _rstr = "確認できません"
                # collect unique 放棄Zoiper in order of appearance
                _seen: dict[str, None] = {}
                for _mc in _miss_list:
                    for _zk in _mc.get("放棄Zoiper", "-").split(", "):
                        _zk = _zk.strip()
                        if _zk and _zk != "-" and _zk in _zoiper_name:
                            _seen[_zk] = None
                _zoipers = list(_seen.keys())
                _rs_count = max(len(_zoipers), 1)
                _hlabel = f'{_h}時台（{len(_miss_list)}件）'
                if not _zoipers:
                    _det += (
                        f'<tr><td>{_hlabel}</td><td>{_oh}</td>'
                        f'<td>-</td>'
                        f'<td style="text-align:left">{_rstr}</td></tr>\n'
                    )
                else:
                    for _zi, _zk in enumerate(_zoipers):
                        _zname = _zoiper_name.get(_zk, "")
                        _zlabel = f'{_zk} {_zname}' if _zname else _zk
                        if _zi == 0:
                            _det += (
                                f'<tr>'
                                f'<td rowspan="{_rs_count}">{_hlabel}</td>'
                                f'<td rowspan="{_rs_count}">{_oh}</td>'
                                f'<td style="text-align:left">{_zlabel}</td>'
                                f'<td rowspan="{_rs_count}" style="text-align:left">{_rstr}</td>'
                                f'</tr>\n'
                            )
                        else:
                            _det += (
                                f'<tr>'
                                f'<td style="text-align:left">{_zlabel}</td>'
                                f'</tr>\n'
                            )
            _det += '</table>'
            ab_ranking_html += (
                f'<tr id="{_did}" style="display:none">'
                f'<td colspan="7" style="padding:8px 8px 8px 24px;background:#f9f9f9">{_det}</td>'
                f'</tr>\n'
            )
        ab_ranking_html += '</table>\n</div>\n'
    else:
        ab_ranking_html = ""

    # 通知用：プレーンテキスト（HTMLツールチップを含めない）。月=前月同曜日平均比, 週=前週同曜日比
    total_month_d,  total_week_d  = dcell_plain(d["total"],         "total",    "件")[:2]
    got_month_d,    got_week_d    = dcell_plain(d["got"],            "got",      "件")[:2]
    missed_month_d, missed_week_d = dcell_plain(d["missed"],         "missed",   "件")[:2]
    rate_month_d,   rate_week_d   = dcell_plain(d["rate"],           "rate",     "%",  True)[:2]
    work_month_d,   work_week_d   = dcell_plain(d["total_work_hrs"], "work_hrs", "h",  False, True)[:2]
    cph_month_d,    cph_week_d    = dcell_plain(cph_raw,             "avg_cph",  "",   False, True)[:2]
    talk_month_d,   talk_week_d   = dcell_plain(talk_raw,            "avg_talk", "秒")[:2]

    def _ext_row_dict(row) -> dict:
        return {
            "ext":    str(row["内線番号"]),
            "name":   str(row["案件名"]),
            "type":   str(row["内線種別"]),
            "total":  int(row["着信"]),
            "got":    int(row["応答"]),
            "missed": int(row["未応答"]),
            "rate":   float(row["応答率"]),
        }

    def _op_row_dict(row) -> dict:
        return {
            "zoiper": str(row["Zoiper"]),
            "name":   str(row["OP名"]),
            "total":  int(row["着信"]),
            "got":    int(row["応答"]),
            "missed": int(row["未応答"]),
            "rate":   (float(row["応答率"]) if row["応答率"] is not None else None),
            "cph":    (float(row["CPH"]) if row["CPH"] is not None else None),
        }

    factors_data = _compute_factors(d)

    _rdata_json = json.dumps({
        "date": d["date_csv"].replace("/", "-"),
        "weekday": d["weekday"],
        "total": d["total"],
        "got": d["got"],
        "missed": d["missed"],
        "rate": d["rate"],
        "work_hrs": round(float(d["total_work_hrs"]), 1),
        "avg_cph": d["avg_cph"],
        "avg_talk_secs": d["avg_talk_secs"],
        "kpi_diffs": {
            "total_month":  total_month_d,  "total_week":  total_week_d,
            "got_month":    got_month_d,    "got_week":    got_week_d,
            "missed_month": missed_month_d, "missed_week": missed_week_d,
            "rate_month":   rate_month_d,   "rate_week":   rate_week_d,
            "work_month":   work_month_d,   "work_week":   work_week_d,
            "cph_month":    cph_month_d,    "cph_week":    cph_week_d,
            "talk_month":   talk_month_d,   "talk_week":   talk_week_d,
        },
        "ext_worst5": [
            _ext_row_dict(row)
            for _, row in d["ext_worst5"].iterrows()
            if row["応答率"] < 100
        ],
        "ext_special": [
            _ext_row_dict(row)
            for _, row in d["ext_special"].iterrows()
        ],
        "op_cph_worst5": [
            _op_row_dict(row)
            for _, row in d["op_cph_worst5"].iterrows()
        ],
        "ext_daily": d["ext_daily"],
        "op_daily": d["op_daily"],
        "factors": factors_data,
        "hourly": [
            {"h": i, "total": r["着信"], "got": r["応答"], "missed": r["未応答"],
             "rate": r["応答率"], "ops": r["active_ops"], "avg_talk": r.get("avg_talk_secs")}
            for i, r in enumerate(d["hourly"])
        ],
    }, ensure_ascii=False)

    factors_html = build_factors_html(d)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>コールセンター日次分析 {d['date_csv']}</title>
{_CSS}
</head>
<body>
<div class="search-bar">
<label for="search-input">🔍 検索：</label>
<input id="search-input" type="text" placeholder="案件名・内線番号・OP名..." oninput="searchFilter(this.value)">
<span class="search-count" id="search-count"></span>
</div>
<h1>{d['date_csv']}（{d['weekday']}）</h1>
<h2>全体サマリー</h2>
<div class="side-by-side">
<div>{_html_table(["KPI", "実績", "前月同曜日平均比", "前週同曜日比"], summary_rows)}</div>
<div><h3>未応答分類（計{d['missed']}件）</h3>
{_html_table(["分類", "件数", "割合"], miss_rows)}</div>
</div>
<h2>要因分析</h2>
{factors_html}
<h2>時間帯別応答率</h2>
{hourly_table_html}
<div id="ranking-section">
<h2>内線別ランキング</h2>
<div class="side-by-side">
<div><h3>着信数ランキング</h3>
{_html_table(ext_headers, _ext_rows_html(d['ext_by_volume']))}</div>
<div><h3>通話時間ランキング</h3>
{talk_section_html}</div>
</div>
<div class="side-by-side">
<div><h4>応答率ベスト5</h4>
{_html_table(ext_headers, _ext_rows_html(d['ext_best5']))}</div>
<div><h4>応答率ワースト5</h4>
{_html_table(ext_headers, _ext_rows_html(d['ext_worst5']))}</div>
</div>
{ab_ranking_html}
<h2>OP別ランキング</h2>
<div class="side-by-side">
<div><h3>CPHベスト5</h3>
{_html_table(op_cph_headers, _op_cph_rows_html(d['op_cph_best5']))}</div>
<div><h3>CPHワースト5</h3>
{_html_table(op_cph_headers, _op_cph_rows_html(d['op_cph_worst5']))}</div>
</div>
<div class="side-by-side">
<div><h3>応答率ベスト5</h3>
{_html_table(op_rate_headers, _op_rate_rows_html(d['op_rate_best5']))}</div>
<div><h3>応答率ワースト5</h3>
{_html_table(op_rate_headers, _op_rate_rows_html(d['op_rate_worst5']))}</div>
</div>
</div>
<script>
function toggleDetail(h) {{
  var row = document.getElementById('detail-' + h);
  var hdr = document.getElementById('arrow-' + h);
  if (!row) return;
  var open = row.style.display !== 'none';
  var tbl = row.closest('table');
  if (tbl) {{
    tbl.querySelectorAll('.detail-row').forEach(function(r) {{
      if (r !== row) r.style.display = 'none';
    }});
    tbl.querySelectorAll('.hour-hdr').forEach(function(el) {{
      if (el !== hdr) el.dataset.open = '';
    }});
  }}
  row.style.display = open ? 'none' : 'table-row';
  if (hdr) hdr.dataset.open = open ? '' : '1';
}}
function toggleAbRow(rowId, detId) {{
  var row = document.getElementById(rowId);
  var det = document.getElementById(detId);
  if (!row || !det) return;
  var open = row.dataset.open === '1';
  var tbl = row.closest('table');
  if (tbl) {{
    tbl.querySelectorAll('.ab-row').forEach(function(r) {{ r.dataset.open = ''; }});
    tbl.querySelectorAll('[id^="ab-det-"]').forEach(function(d) {{ d.style.display = 'none'; }});
  }}
  if (!open) {{
    row.dataset.open = '1';
    det.style.display = 'table-row';
  }}
}}
function searchFilter(q) {{
  q = q.trim().toLowerCase();
  var section = document.getElementById('ranking-section');
  if (!section) return;
  var count = 0;
  section.querySelectorAll('table').forEach(function(tbl) {{
    Array.from(tbl.rows).slice(1).forEach(function(tr) {{
      var match = !q || tr.textContent.toLowerCase().includes(q);
      tr.style.display = match ? '' : 'none';
      if (match && q) count++;
    }});
  }});
  var el = document.getElementById('search-count');
  if (el) el.textContent = q ? count + '件ヒット' : '';
}}
</script>
<script type="application/json" id="rdata">{_rdata_json}</script>
</body>
</html>"""


# ---- dashboard builder -----------------------------------------------

_DASHBOARD_CSS = """<style>
body{margin:0;padding:0}
#dashboard-wrapper{display:flex;height:100vh;overflow:hidden}
#sidebar{width:210px;min-width:210px;background:#f5f5f5;border-right:2px solid #ddd;padding:12px;overflow-y:auto;flex-shrink:0;transition:width 0.2s,min-width 0.2s,padding 0.2s}
#sidebar.collapsed{width:32px;min-width:32px;padding:6px 4px;overflow:hidden}
#sidebar.collapsed>*:not(#sidebar-toggle){display:none}
#sidebar.collapsed #sidebar-toggle{text-align:center}
#sidebar-toggle{display:block;width:100%;text-align:right;background:none;border:none;cursor:pointer;font-size:1em;padding:0 0 6px;color:#888;font-family:inherit}
#sidebar-toggle:hover{color:#333}
#main-content{flex:1;overflow-y:auto;padding:2em;min-width:0}
#sidebar-title{font-size:0.9em;font-weight:bold;color:#333;margin-bottom:10px;padding-bottom:6px;border-bottom:2px solid #ccc}
#gs-wrap{margin-bottom:12px}
#gs-wrap input{width:100%;box-sizing:border-box;padding:5px 8px;border:1px solid #aaa;border-radius:4px;font-size:0.82em;font-family:inherit}
.cal-nav{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.cal-nav button{background:#fff;border:1px solid #ccc;border-radius:4px;padding:3px 10px;cursor:pointer;font-size:0.9em}
.cal-nav button:hover{background:#e0e0e0}
#month-label{font-weight:bold;font-size:0.95em}
#calendar-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:2px}
.cal-header{text-align:center;font-size:0.7em;font-weight:bold;color:#888;padding:3px 0}
.cal-cell{text-align:center;padding:3px 1px;border-radius:3px;font-size:0.76em;color:#ccc;line-height:1.3}
.cal-available{background:#d4edda;color:#155724;cursor:pointer;font-weight:bold}
.cal-available:hover{background:#b8dacc}
.cal-selected{background:#0d6efd !important;color:#fff !important}
.cal-today{outline:2px solid #0d6efd;outline-offset:-2px}
.report-section{display:none}
.report-section .search-bar{display:none}
#no-selection{color:#aaa;padding:3em;text-align:center;font-size:1.1em}
#search-results-panel{display:none;padding:0.5em 0}
.sr-summary{font-size:0.9em;color:#555;margin-bottom:1em}
.sr-date{font-weight:bold;color:#0d6efd;cursor:pointer;padding:8px 0 4px;border-top:2px solid #ddd;margin-top:12px;font-size:0.95em}
.sr-date:hover{text-decoration:underline}
.sr-section{font-size:0.82em;color:#888;padding:4px 0 2px;margin-top:4px}
.view-section{display:none}
.nav-group{margin-top:10px;padding-top:10px;border-top:1px solid #ddd}
.nav-label{font-size:0.72em;font-weight:bold;color:#888;letter-spacing:0.04em;margin-bottom:4px}
.nav-btn{display:block;width:100%;text-align:left;background:none;border:none;padding:5px 6px;cursor:pointer;font-size:0.82em;border-radius:4px;color:#333;font-family:inherit;margin-bottom:2px}
.nav-btn:hover{background:#e0e8ff}
.nav-btn.active{background:#0d6efd;color:#fff}
.month-tabs{display:flex;flex-wrap:wrap;gap:4px;margin:12px 0 0;border-bottom:2px solid #dee2e6;padding-bottom:0}
.mtab{background:none;border:1px solid transparent;border-bottom:none;padding:5px 12px;cursor:pointer;font-size:0.82em;border-radius:4px 4px 0 0;color:#555;font-family:inherit;margin-bottom:-2px}
.mtab:hover{background:#e8f0fe;color:#1a73e8}
.mtab.mtab-active{background:#fff;border-color:#dee2e6;border-bottom-color:#fff;color:#0d6efd;font-weight:bold}
.mtab-panel{padding-top:12px;max-height:75vh;overflow:auto}
.mtab-panel table{font-size:0.8em;white-space:nowrap;border-collapse:separate;border-spacing:0}
.mtab-panel th,.mtab-panel td{border:1px solid #ccc;padding:3px 6px;text-align:center}
.mtab-panel thead th{position:sticky;top:0;z-index:3;background:#f0f0f0}
.sticky-l1{position:sticky;left:0;z-index:2;background:#fafafa;min-width:36px}
.sticky-l2{position:sticky;left:36px;z-index:2;background:#fafafa;min-width:90px;text-align:left}
.sticky-l3{position:sticky;left:126px;z-index:2;background:#fafafa;min-width:54px;text-align:left}
thead th.sticky-l1,thead th.sticky-l2,thead th.sticky-l3{z-index:4}
.metric-lbl{font-size:0.75em;color:#444;text-align:left}
.total-block td,.total-block th{background:#f5f5f5;font-weight:bold}
.row-grp-first td{border-top:2px solid #aaa}
.sort-th{cursor:pointer;user-select:none}
thead .sort-th:hover{background:#dde8fc}
#legend-wrap{margin-top:12px;padding-top:10px;border-top:1px solid #ddd;font-size:0.78em}
.legend-title{font-weight:bold;color:#555;margin-bottom:5px}
.legend-cat{font-weight:bold;color:#666;margin:6px 0 2px;font-size:0.9em}
.legend-row{display:flex;align-items:center;gap:5px;margin:2px 0}
.legend-swatch{display:inline-block;width:11px;height:11px;border-radius:2px;border:1px solid #ccc;flex-shrink:0}
.legend-text{color:#444}
</style>"""


_VIEWS_JS = """
var currentViewId = null;
var viewFrom = '';
var viewTo   = '';
var extFilterType = '';
var extFilterExt  = '';
var _mVData = {};
var _mTSort = {};
var _mTSortDef = {
  'op-day':{key:'cph',dir:'desc'},
  'op-hour':{key:'cph',dir:'desc'},
  'op-weekday':{key:'cph',dir:'desc'},
  'op-type':{key:'cph',dir:'desc'}
};

function fmtSecs(s) {
  if (s == null) return '-';
  s = Math.round(s);
  return Math.floor(s / 60) + '分' + String(s % 60).padStart(2, '0') + '秒';
}
function rateCls(r) {
  if (r == null) return '';
  return r >= 90 ? 'green' : r >= 80 ? 'yellow' : 'red';
}
function cphCls(c) {
  if (c == null) return '';
  return c >= 10 ? 'green' : c >= 8 ? 'yellow' : 'red';
}
function getViewData() {
  if (!viewFrom && !viewTo) return dailyData;
  return dailyData.filter(function(r) {
    if (viewFrom && r.date < viewFrom) return false;
    if (viewTo   && r.date > viewTo)   return false;
    return true;
  });
}
function getFilteredDays() {
  var days = getViewData();
  if (!extFilterType && !extFilterExt) return days;
  return days.map(function(d) {
    var exts = (d.ext_daily || []).filter(function(e) {
      if (extFilterExt)   return e.ext  === extFilterExt;
      if (extFilterType)  return e.type === extFilterType;
      return true;
    });
    if (!exts.length) return null;
    var tot = exts.reduce(function(s,e){return s+e.total;},0);
    var got = exts.reduce(function(s,e){return s+e.got;},0);
    var mis = tot - got;
    var rate = tot > 0 ? Math.round(got/tot*1000)/10 : null;
    var hourly = [];
    for (var h=0; h<24; h++) {
      var hT=0, hG=0;
      exts.forEach(function(e){hT+=(e.h&&e.h[h])||0; hG+=(e.g&&e.g[h])||0;});
      hourly.push({total:hT, got:hG, missed:hT-hG,
        rate: hT>0 ? Math.round(hG/hT*1000)/10 : null, ops:null, avg_talk:null});
    }
    return {date:d.date, weekday:d.weekday, total:tot, got:got, missed:mis, rate:rate,
            work_hrs:null, avg_cph:null, avg_talk_secs:null, hourly:hourly};
  }).filter(Boolean);
}
function applyViewFilter() {
  var ff = document.getElementById('filter-from');
  var ft = document.getElementById('filter-to');
  viewFrom = ff ? ff.value : '';
  viewTo   = ft ? ft.value : '';
  var cnt = document.getElementById('filter-count');
  if (cnt) cnt.textContent = (viewFrom || viewTo) ? getViewData().length + '日分' : '';
  if (!currentViewId) return;
  var el = document.getElementById('view-' + currentViewId);
  if (el) renderViewContent(currentViewId, el);
}
function clearViewFilter() {
  viewFrom = ''; viewTo = '';
  var ff = document.getElementById('filter-from');
  var ft = document.getElementById('filter-to');
  var cnt = document.getElementById('filter-count');
  if (ff) ff.value = '';
  if (ft) ft.value = '';
  if (cnt) cnt.textContent = '';
  clearExtFilter();
  if (!currentViewId) return;
  var el = document.getElementById('view-' + currentViewId);
  if (el) renderViewContent(currentViewId, el);
}
function setFilterDates(from, to) {
  viewFrom = from; viewTo = to;
  var ff = document.getElementById('filter-from');
  var ft = document.getElementById('filter-to');
  if (ff) ff.value = from;
  if (ft) ft.value = to;
}
function buildExtDropdowns() {
  var typeEl = document.getElementById('ext-type-filter');
  var numEl  = document.getElementById('ext-num-filter');
  if (!typeEl || !numEl) return;
  var typeSet = {};
  var extMap  = {};
  dailyData.forEach(function(d) {
    (d.ext_daily || []).forEach(function(e) {
      typeSet[e.type] = true;
      if (!extMap[e.ext]) extMap[e.ext] = {name: e.name, type: e.type};
    });
  });
  var types = Object.keys(typeSet).sort(function(a,b){return Number(a)-Number(b)||a.localeCompare(b);});
  typeEl.innerHTML = '<option value="">種別: 全体</option>';
  types.forEach(function(t) {
    var opt = document.createElement('option');
    opt.value = t; opt.textContent = '種別 ' + t;
    if (t === extFilterType) opt.selected = true;
    typeEl.appendChild(opt);
  });
  refreshExtNumDropdown(extMap);
}
function refreshExtNumDropdown(extMap) {
  var numEl  = document.getElementById('ext-num-filter');
  var typeEl = document.getElementById('ext-type-filter');
  if (!numEl) return;
  if (!extMap) {
    extMap = {};
    dailyData.forEach(function(d) {
      (d.ext_daily || []).forEach(function(e) {
        if (!extMap[e.ext]) extMap[e.ext] = {name: e.name, type: e.type};
      });
    });
  }
  var selType = typeEl ? typeEl.value : extFilterType;
  var exts = Object.keys(extMap).sort();
  numEl.innerHTML = '<option value="">内線: 全体</option>';
  exts.forEach(function(ext) {
    var info = extMap[ext];
    if (selType && info.type !== selType) return;
    var opt = document.createElement('option');
    opt.value = ext;
    var label = ext + '  ' + (info.name.length > 22 ? info.name.substring(0,22)+'…' : info.name);
    opt.textContent = label;
    if (ext === extFilterExt) opt.selected = true;
    numEl.appendChild(opt);
  });
}
function applyExtFilter() {
  var typeEl = document.getElementById('ext-type-filter');
  var numEl  = document.getElementById('ext-num-filter');
  var newType = typeEl ? typeEl.value : '';
  var newExt  = numEl  ? numEl.value  : '';
  if (newType !== extFilterType && newExt) {
    var extMap = {};
    dailyData.forEach(function(d) {
      (d.ext_daily||[]).forEach(function(e){if(!extMap[e.ext])extMap[e.ext]={type:e.type};});
    });
    var info = extMap[newExt];
    if (info && newType && info.type !== newType) {
      newExt = '';
      if (numEl) numEl.value = '';
    }
  }
  extFilterType = newType;
  extFilterExt  = newExt;
  refreshExtNumDropdown(null);
  if (!currentViewId) return;
  var el = document.getElementById('view-' + currentViewId);
  if (el) renderViewContent(currentViewId, el);
}
function clearExtFilter() {
  extFilterType = ''; extFilterExt = '';
  var typeEl = document.getElementById('ext-type-filter');
  var numEl  = document.getElementById('ext-num-filter');
  if (typeEl) typeEl.value = '';
  if (numEl)  numEl.value  = '';
}
function showView(viewId) {
  var gsInput = document.getElementById('gs-input');
  if (gsInput) gsInput.value = '';
  var srPanel = document.getElementById('search-results-panel');
  srPanel.style.display = 'none'; srPanel.innerHTML = '';
  document.querySelectorAll('.report-section,.view-section').forEach(function(el) { el.style.display = 'none'; });
  document.getElementById('no-selection').style.display = 'none';
  var fb = document.getElementById('view-filter-bar');
  if (fb) fb.style.display = 'flex';
  buildExtDropdowns();
  selectedDate = null;
  renderCalendar();
  document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });
  var btn = document.querySelector('[data-view="' + viewId + '"]');
  if (btn) btn.classList.add('active');
  currentViewId = viewId;
  if (viewId === 'prev-month' || viewId === 'this-month') {
    var today = new Date();
    var y, m;
    if (viewId === 'this-month') { y = today.getFullYear(); m = today.getMonth() + 1; }
    else { var prev = new Date(today.getFullYear(), today.getMonth() - 1, 1); y = prev.getFullYear(); m = prev.getMonth() + 1; }
    var pad = function(n) { return String(n).padStart(2, '0'); };
    var firstDay = y + '-' + pad(m) + '-01';
    var lastDate = new Date(y, m, 0).getDate();
    var lastDay  = y + '-' + pad(m) + '-' + pad(lastDate);
    setFilterDates(firstDay, lastDay);
    var cnt = document.getElementById('filter-count');
    if (cnt) cnt.textContent = getViewData().length + '日分';
  }
  var el = document.getElementById('view-' + viewId);
  if (!el) return;
  el.style.display = 'block';
  document.getElementById('main-content').scrollTop = 0;
  renderViewContent(viewId, el);
}
function renderViewContent(id, el) {
  if (id === 'prev-month' || id === 'this-month') renderMonthView(el, id === 'this-month');
  else if (id === 'by-weekday') renderWeekdayView(el);
  else if (id === 'by-hour')    renderHourView(el);
}
function extFilterLabel() {
  if (extFilterExt)  return '内線 ' + extFilterExt;
  if (extFilterType) return '種別 ' + extFilterType;
  return '';
}
function switchMTab(btn, viewId, tabId) {
  var prefix = 'mtab-' + viewId;
  btn.closest('.month-tabs').querySelectorAll('.mtab').forEach(function(b) { b.classList.remove('mtab-active'); });
  btn.classList.add('mtab-active');
  var wrap = btn.closest('.view-section');
  wrap.querySelectorAll('.mtab-panel').forEach(function(p) { p.style.display = 'none'; });
  var panel = document.getElementById(prefix + '-' + tabId);
  if (panel) panel.style.display = 'block';
}
function mTabSort(viewId, tabId, sortKey) {
  var sk = viewId + '-' + tabId;
  var cur = _mTSort[sk] || _mTSortDef[tabId] || {key:null, dir:'asc'};
  _mTSort[sk] = {key:sortKey, dir:(cur.key===sortKey&&cur.dir==='asc')?'desc':'asc'};
  var panel = document.getElementById('mtab-' + viewId + '-' + tabId);
  if (panel) panel.innerHTML = buildMTabContent(tabId, viewId, _mVData[viewId]||[]);
}

function renderMonthView(el, isThis) {
  var viewId = isThis ? 'this-month' : 'prev-month';
  var today = new Date();
  var y, m;
  if (isThis) { y = today.getFullYear(); m = today.getMonth() + 1; }
  else {
    var prev = new Date(today.getFullYear(), today.getMonth() - 1, 1);
    y = prev.getFullYear(); m = prev.getMonth() + 1;
  }
  var title = (isThis ? '当月累計' : '前月累計') + ' (' + y + '年' + m + '月)';
  var rows = getFilteredDays();
  rows.sort(function(a, b) { return a.date.localeCompare(b.date); });
  _mVData[viewId] = rows;
  if (!rows.length) {
    el.innerHTML = '<h2>' + title + '</h2><p style="color:#aaa">データなし</p>';
    return;
  }
  var period = rows[0].date + ' 〜 ' + rows[rows.length - 1].date;
  var fl = extFilterLabel();
  var ts = fl ? ' <span style="font-size:0.75em;background:#e8f0fe;color:#1a73e8;border-radius:3px;padding:2px 6px">' + fl + '</span>' : '';
  var html = '<h2>' + title + ts + '</h2>';
  html += '<p style="color:#777;font-size:0.9em">対象期間: ' + period + '（' + rows.length + '日）</p>';
  if (fl) html += '<p style="color:#888;font-size:0.85em">※ 案件絞り込み中。稼働時間・CPHは全体値です。</p>';

  var tabs = [
    {id:'type-day',      label:'種別×日'},
    {id:'type-hour',     label:'種別×時間'},
    {id:'type-weekday',  label:'種別×曜日'},
    {id:'day-hour',      label:'日×時間'},
    {id:'weekday-hour',  label:'曜日×時間'},
    {id:'op-day',        label:'Zoiper×日'},
    {id:'op-hour',       label:'Zoiper×時間'},
    {id:'op-weekday',    label:'Zoiper×曜日'},
    {id:'op-type',       label:'Zoiper×種別'},
  ];
  html += '<div class="month-tabs">';
  tabs.forEach(function(t, i) {
    html += `<button class="mtab${i===0?' mtab-active':''}" onclick="switchMTab(this,'${viewId}','${t.id}')">${t.label}</button>`;
  });
  html += '</div>';
  tabs.forEach(function(t, i) {
    html += '<div class="mtab-panel" id="mtab-' + viewId + '-' + t.id + '" style="display:' + (i===0?'block':'none') + '">';
    html += buildMTabContent(t.id, viewId, rows);
    html += '</div>';
  });
  el.innerHTML = html;
}

function buildMTabContent(tabId, viewId, rows) {
  if (tabId === 'type-day')     return buildTypeDayTab(rows, viewId, tabId);
  if (tabId === 'type-hour')    return buildTypeHourTab(rows, viewId, tabId);
  if (tabId === 'type-weekday') return buildTypeWeekdayTab(rows, viewId, tabId);
  if (tabId === 'day-hour')     return buildDayHourTab(rows, viewId, tabId);
  if (tabId === 'weekday-hour') return buildWeekdayHourTab(rows, viewId, tabId);
  if (tabId === 'op-day')       return buildOpDayTab(rows, viewId, tabId);
  if (tabId === 'op-hour')      return buildOpHourTab(rows, viewId, tabId);
  if (tabId === 'op-weekday')   return buildOpWeekdayTab(rows, viewId, tabId);
  if (tabId === 'op-type')      return buildOpTypeTab(rows, viewId, tabId);
  return '';
}

function _typeName(t) { return typeNames[t] || t; }
function _rateCell(got, total) {
  if (!total) return '<td style="color:#ccc">-</td>';
  var rate = Math.round(got/total*1000)/10;
  return '<td class="' + rateCls(rate) + '" title="' + rate + '%">' + got + '/' + total + '</td>';
}
function _cphVal(got, wh) { return (wh > 0 && got > 0) ? Math.round(got/wh*10)/10 : null; }
function _fmtW(w) { return w > 0 ? w.toFixed(1) + 'h' : '-'; }
function _fmtTalk(s) { return s != null ? Math.floor(s/60) + ':' + ('0'+s%60).slice(-2) : '-'; }

// 絞り込み中は ext_daily から合計を再計算
function _getFilteredTotals(r) {
  if (!extFilterType && !extFilterExt) {
    return {total:r.total, got:r.got, missed:r.missed,
            work_hrs:r.work_hrs||0, avg_cph:r.avg_cph, avg_talk_secs:r.avg_talk_secs};
  }
  var t=0, g=0, ts=0;
  (r.ext_daily||[]).forEach(function(e) {
    if (extFilterType && e.type !== extFilterType) return;
    if (extFilterExt  && e.ext  !== extFilterExt)  return;
    t += e.total; g += e.got; ts += e.talk_sum||0;
  });
  return {total:t, got:g, missed:t-g,
          work_hrs:r.work_hrs||0, avg_cph:r.avg_cph,
          avg_talk_secs: g>0 ? Math.round(ts/g) : null};
}

// 種別×* テーブルヘッダー（#, 種別名, 指標名 の3固定列）
// sort header helper — uses template literal so onclick quotes are safe
function _sth(lbl, sk, cls, vid, tid, spec) {
  var arrow = spec&&spec.key===sk ? (spec.dir==='desc'?'▼':'▲') : '';
  var c = cls ? cls+' sort-th' : 'sort-th';
  return `<th class="${c}" onclick="mTabSort('${vid}','${tid}','${sk}')">${lbl}${arrow?' '+arrow:''}</th>`;
}

function _typeTableHead(colLabels, colKeys, viewId, tabId, spec) {
  var h = '<thead><tr>';
  h += _sth('#','num','sticky-l1',viewId,tabId,spec);
  h += _sth('種別','name','sticky-l2',viewId,tabId,spec);
  h += '<th class="sticky-l3">指標</th>';
  colLabels.forEach(function(l,i){ h += _sth(l,colKeys[i],'',viewId,tabId,spec); });
  h += _sth('合計','total','',viewId,tabId,spec);
  h += '</tr></thead>';
  return h;
}

// 種別の4指標行（着信/応答/未応答/応答率）を生成
function _typeRows(typeNo, typeName, metrics, cols) {
  // metrics: {total, got, missed, colData:[{total,got},...]}
  var LABELS = ['着信数','応答数','未応答数','応答率'];
  var html = '';
  LABELS.forEach(function(lbl, li) {
    var cls = li===0 ? ' class="row-grp-first"' : '';
    html += '<tr' + cls + '>';
    if (li===0) {
      html += '<td class="sticky-l1" rowspan="4">' + typeNo + '</td>';
      html += '<td class="sticky-l2" rowspan="4">' + typeName + '</td>';
    }
    html += '<td class="sticky-l3 metric-lbl">' + lbl + '</td>';
    cols.forEach(function(c) {
      if (li===0) html += '<td>' + (c.total||0) + '</td>';
      else if (li===1) html += '<td>' + (c.got||0) + '</td>';
      else if (li===2) html += '<td>' + ((c.total||0)-(c.got||0)) + '</td>';
      else {
        if (!c.total) html += '<td style="color:#ccc">-</td>';
        else { var r=Math.round(c.got/c.total*1000)/10; html += '<td class="'+rateCls(r)+'" title="'+r+'%">'+r+'%</td>'; }
      }
    });
    // 合計列
    if (li===0) html += '<td>' + metrics.total + '</td>';
    else if (li===1) html += '<td>' + metrics.got + '</td>';
    else if (li===2) html += '<td>' + metrics.missed + '</td>';
    else {
      if (!metrics.total) html += '<td style="color:#ccc">-</td>';
      else { var r=Math.round(metrics.got/metrics.total*1000)/10; html += '<td class="'+rateCls(r)+'" title="'+r+'%">'+r+'%</td>'; }
    }
    html += '</tr>';
  });
  return html;
}

// 合計ブロック行（種別×*）: 4指標
function _typeTotalBlock(colAgg, grandTotal, grandGot) {
  var LABELS = ['着信数','応答数','未応答数','応答率'];
  var html = '';
  LABELS.forEach(function(lbl, li) {
    var cls = li===0 ? ' class="total-block row-grp-first"' : ' class="total-block"';
    html += '<tr' + cls + '>';
    if (li===0) {
      html += '<td class="sticky-l1 total-block" rowspan="4">合計</td>';
      html += '<td class="sticky-l2 total-block" rowspan="4"></td>';
    }
    html += '<td class="sticky-l3 metric-lbl">' + lbl + '</td>';
    colAgg.forEach(function(c) {
      if (li===0) html += '<td>' + c.total + '</td>';
      else if (li===1) html += '<td>' + c.got + '</td>';
      else if (li===2) html += '<td>' + (c.total-c.got) + '</td>';
      else {
        if (!c.total) html += '<td style="color:#ccc">-</td>';
        else { var r=Math.round(c.got/c.total*1000)/10; html += '<td class="'+rateCls(r)+'">'+r+'%</td>'; }
      }
    });
    // 右端合計
    if (li===0) html += '<td>' + grandTotal + '</td>';
    else if (li===1) html += '<td>' + grandGot + '</td>';
    else if (li===2) html += '<td>' + (grandTotal-grandGot) + '</td>';
    else {
      if (!grandTotal) html += '<td style="color:#ccc">-</td>';
      else { var r=Math.round(grandGot/grandTotal*1000)/10; html += '<td class="'+rateCls(r)+'">'+r+'%</td>'; }
    }
    html += '</tr>';
  });
  return html;
}

// Zoiper×* テーブルヘッダー（Zoiper, OP名, 指標名 の3固定列）
function _opTableHead(colLabels, colKeys, viewId, tabId, spec) {
  var h = '<thead><tr>';
  h += _sth('Zoiper','zoiper','sticky-l1',viewId,tabId,spec);
  h += _sth('OP名','name','sticky-l2',viewId,tabId,spec);
  h += '<th class="sticky-l3">指標</th>';
  colLabels.forEach(function(l,i){ h += _sth(l,colKeys[i],'',viewId,tabId,spec); });
  h += _sth('合計','total','',viewId,tabId,spec);
  h += '</tr></thead>';
  return h;
}
function _dayHourHead(colLabels, colKeys, viewId, tabId, spec) {
  var h = '<thead><tr>';
  h += _sth('日付','date','sticky-l1',viewId,tabId,spec);
  h += _sth('曜日','weekday','sticky-l2',viewId,tabId,spec);
  h += '<th class="sticky-l3">指標</th>';
  colLabels.forEach(function(l,i){ h += _sth(l,colKeys[i],'',viewId,tabId,spec); });
  h += _sth('合計','total','',viewId,tabId,spec);
  h += '</tr></thead>';
  return h;
}
function _wdHourHead(colLabels, colKeys, viewId, tabId, spec) {
  var h = '<thead><tr>';
  h += _sth('曜日','weekday','sticky-l1',viewId,tabId,spec);
  h += _sth('日数','cnt','sticky-l2',viewId,tabId,spec);
  h += '<th class="sticky-l3">指標</th>';
  colLabels.forEach(function(l,i){ h += _sth(l,colKeys[i],'',viewId,tabId,spec); });
  h += _sth('合計','total','',viewId,tabId,spec);
  h += '</tr></thead>';
  return h;
}
function _sortTypes(types, spec, typeNums, typeData) {
  var k=(spec&&spec.key)||'num', d=spec&&spec.dir==='desc'?-1:1;
  types.sort(function(a,b){
    if(k==='num') return d*((typeNums[a]||0)-(typeNums[b]||0));
    if(k==='name') return d*_typeName(a).localeCompare(_typeName(b));
    if(k==='total') return d*((typeData[a].total||0)-(typeData[b].total||0));
    var mi=k.match(/^col_([0-9]+)$/);
    if(mi){var ci=+mi[1];return d*((typeData[a].colV[ci]||0)-(typeData[b].colV[ci]||0));}
    return 0;
  });
}
function _sortOps(ops, spec, opData, opNames) {
  var k=(spec&&spec.key)||'cph', d=spec&&spec.dir==='desc'?-1:1;
  ops.sort(function(a,b){
    var va, vb;
    if(k==='cph') {
      va=_cphVal(opData[a].got,opData[a].work_hrs)||0;
      vb=_cphVal(opData[b].got,opData[b].work_hrs)||0;
      return (va-vb)*d || (opData[b].got-opData[a].got);
    }
    if(k==='zoiper') return d*a.localeCompare(b);
    if(k==='name') return d*(opNames[a]||'').localeCompare(opNames[b]||'');
    if(k==='total'){va=opData[a].total||0;vb=opData[b].total||0;return d*(va-vb);}
    if(k==='got')  {va=opData[a].got||0;  vb=opData[b].got||0;  return d*(va-vb);}
    var mi=k.match(/^col_([0-9]+)$/);
    if(mi){var ci=+mi[1];va=opData[a].colV[ci]||0;vb=opData[b].colV[ci]||0;return d*(va-vb);}
    return 0;
  });
}

// OPの7指標行を生成 (着信/応答/未応答/応答率/稼働時間/CPH/通話時間)
function _opRows(zoiper, opName, totals, cols) {
  // cols: [{total, got}, ...] or [{got}, ...] (通話時間/稼働時間はtotals側のみ)
  var LABELS = ['着信数','応答数','未応答数','応答率','稼働時間','CPH','通話時間'];
  var html = '';
  LABELS.forEach(function(lbl, li) {
    var cls = li===0 ? ' class="row-grp-first"' : '';
    html += '<tr' + cls + '>';
    if (li===0) {
      html += '<td class="sticky-l1" rowspan="7">' + zoiper + '</td>';
      html += '<td class="sticky-l2" rowspan="7">' + opName + '</td>';
    }
    html += '<td class="sticky-l3 metric-lbl">' + lbl + '</td>';
    cols.forEach(function(c) {
      if (li===0) html += '<td>' + (c.total!=null?c.total:'-') + '</td>';
      else if (li===1) html += '<td>' + (c.got!=null?c.got:'-') + '</td>';
      else if (li===2) html += '<td>' + (c.missed!=null?c.missed:(c.total!=null&&c.got!=null?c.total-c.got:'-')) + '</td>';
      else if (li===3) {
        if (!c.total) html += '<td style="color:#ccc">-</td>';
        else { var r=Math.round(c.got/c.total*1000)/10; html += '<td class="'+rateCls(r)+'">'+r+'%</td>'; }
      }
      else html += '<td style="color:#ccc">-</td>';
    });
    // 合計列
    var tot=totals;
    if (li===0) html += '<td>' + (tot.total!=null?tot.total:'-') + '</td>';
    else if (li===1) html += '<td>' + (tot.got!=null?tot.got:'-') + '</td>';
    else if (li===2) html += '<td>' + (tot.missed!=null?tot.missed:'-') + '</td>';
    else if (li===3) {
      if (!tot.total) html += '<td style="color:#ccc">-</td>';
      else { var r=Math.round(tot.got/tot.total*1000)/10; html += '<td class="'+rateCls(r)+'">'+r+'%</td>'; }
    }
    else if (li===4) html += '<td>' + _fmtW(tot.work_hrs||0) + '</td>';
    else if (li===5) { var cph=_cphVal(tot.got,tot.work_hrs||0); html += '<td class="'+cphCls(cph)+'">'+(cph!=null?cph:'-')+'</td>'; }
    else html += '<td>' + _fmtTalk(tot.avg_talk_secs) + '</td>';
    html += '</tr>';
  });
  return html;
}

// OP合計ブロック（7指標）
function _opTotalBlock(colAgg, grandTotals, extraCells) {
  var LABELS = ['着信数','応答数','未応答数','応答率','稼働時間','CPH','通話時間'];
  var html = '';
  LABELS.forEach(function(lbl, li) {
    var cls = li===0 ? ' class="total-block row-grp-first"' : ' class="total-block"';
    html += '<tr' + cls + '>';
    if (li===0) {
      html += '<td class="sticky-l1 total-block" rowspan="7">合計</td>';
      html += '<td class="sticky-l2 total-block" rowspan="7"></td>';
    }
    html += '<td class="sticky-l3 metric-lbl">' + lbl + '</td>';
    colAgg.forEach(function(c) {
      if (li===0) html += '<td>' + (c.total||0) + '</td>';
      else if (li===1) html += '<td>' + (c.got||0) + '</td>';
      else if (li===2) html += '<td>' + ((c.total||0)-(c.got||0)) + '</td>';
      else if (li===3) {
        if (!c.total) html += '<td style="color:#ccc">-</td>';
        else { var r=Math.round(c.got/c.total*1000)/10; html += '<td class="'+rateCls(r)+'">'+r+'%</td>'; }
      }
      else html += '<td style="color:#ccc">-</td>';
    });
    var gt = grandTotals;
    if (li===0) html += '<td>' + (gt.total||0) + '</td>';
    else if (li===1) html += '<td>' + (gt.got||0) + '</td>';
    else if (li===2) html += '<td>' + ((gt.total||0)-(gt.got||0)) + '</td>';
    else if (li===3) {
      if (!gt.total) html += '<td style="color:#ccc">-</td>';
      else { var r=Math.round(gt.got/gt.total*1000)/10; html += '<td class="'+rateCls(r)+'">'+r+'%</td>'; }
    }
    else if (li===4) html += '<td>' + _fmtW(gt.work_hrs||0) + '</td>';
    else if (li===5) { var cph=_cphVal(gt.got,gt.work_hrs||0); html += '<td class="'+cphCls(cph)+'">'+(cph!=null?cph:'-')+'</td>'; }
    else html += '<td>' + _fmtTalk(gt.avg_talk_secs) + '</td>';
    if (extraCells && li < extraCells.length) html += extraCells[li];
    else if (extraCells) html += '<td></td>';
    html += '</tr>';
  });
  return html;
}

function buildTypeDayTab(rows, viewId, tabId) {
  var types=[],typeSet={},typeNums={};
  rows.forEach(function(r){
    (r.ext_daily||[]).forEach(function(e){
      if(extFilterType&&e.type!==extFilterType)return;
      if(extFilterExt&&e.ext!==extFilterExt)return;
      if(!typeSet[e.type]){typeSet[e.type]=true;types.push(e.type);}
    });
  });
  if(!types.length) return '<p style="color:#aaa">データなし</p>';
  types.sort(function(a,b){return (+a||0)-(+b||0)||a.localeCompare(b);});
  types.forEach(function(t,i){typeNums[t]=i+1;});
  var mat={};
  types.forEach(function(t){mat[t]={};});
  rows.forEach(function(r){
    (r.ext_daily||[]).forEach(function(e){
      if(extFilterType&&e.type!==extFilterType)return;
      if(extFilterExt&&e.ext!==extFilterExt)return;
      if(!mat[e.type])return;
      if(!mat[e.type][r.date])mat[e.type][r.date]={total:0,got:0};
      mat[e.type][r.date].total+=e.total;
      mat[e.type][r.date].got+=e.got;
    });
  });
  var typeData={};
  types.forEach(function(t){
    var rT=0,rG=0;
    var colV=rows.map(function(r){var c=mat[t][r.date]||{total:0,got:0};rT+=c.total;rG+=c.got;return c.total;});
    typeData[t]={total:rT,got:rG,colV:colV};
  });
  var spec=_mTSort[viewId+'-'+tabId]||{key:'num',dir:'asc'};
  _sortTypes(types,spec,typeNums,typeData);
  var colLabels=rows.map(function(r){return r.date.slice(5)+'<br>'+(r.weekday||'');});
  var colKeys=rows.map(function(_,i){return 'col_'+i;});
  var colAgg=rows.map(function(r){var ft=_getFilteredTotals(r);return{total:ft.total,got:ft.got};});
  var gT=colAgg.reduce(function(s,c){return s+c.total;},0);
  var gG=colAgg.reduce(function(s,c){return s+c.got;},0);
  var html='<table>'+_typeTableHead(colLabels,colKeys,viewId,tabId,spec)+'<tbody>';
  html+=_typeTotalBlock(colAgg,gT,gG);
  types.forEach(function(t){
    var td=typeData[t];
    var cols=rows.map(function(r){return mat[t][r.date]||{total:0,got:0};});
    html+=_typeRows(typeNums[t],_typeName(t),{total:td.total,got:td.got,missed:td.total-td.got},cols);
  });
  return html+'</tbody></table>';
}

function buildTypeHourTab(rows, viewId, tabId) {
  var types=[],typeSet={},typeNums={};
  rows.forEach(function(r){
    (r.ext_daily||[]).forEach(function(e){
      if(extFilterType&&e.type!==extFilterType)return;
      if(extFilterExt&&e.ext!==extFilterExt)return;
      if(!typeSet[e.type]){typeSet[e.type]=true;types.push(e.type);}
    });
  });
  if(!types.length) return '<p style="color:#aaa">データなし</p>';
  types.sort(function(a,b){return (+a||0)-(+b||0)||a.localeCompare(b);});
  types.forEach(function(t,i){typeNums[t]=i+1;});
  var mat={};
  types.forEach(function(t){mat[t]=new Array(24).fill(null).map(function(){return{total:0,got:0};});});
  rows.forEach(function(r){
    (r.ext_daily||[]).forEach(function(e){
      if(extFilterType&&e.type!==extFilterType)return;
      if(extFilterExt&&e.ext!==extFilterExt)return;
      if(!mat[e.type])return;
      for(var h=0;h<24;h++){mat[e.type][h].total+=(e.h||[])[h]||0;mat[e.type][h].got+=(e.g||[])[h]||0;}
    });
  });
  var hTots=new Array(24).fill(null).map(function(){return{total:0,got:0};});
  rows.forEach(function(r){for(var h=0;h<24;h++){var hh=r.hourly&&r.hourly[h];if(hh){hTots[h].total+=hh.total||0;hTots[h].got+=hh.got||0;}}});
  var typeData={};
  types.forEach(function(t){
    var rT=0,rG=0;
    var colV=new Array(24).fill(0).map(function(_,h){var c=mat[t][h];rT+=c.total;rG+=c.got;return c.total;});
    typeData[t]={total:rT,got:rG,colV:colV};
  });
  var spec=_mTSort[viewId+'-'+tabId]||{key:'num',dir:'asc'};
  _sortTypes(types,spec,typeNums,typeData);
  var colHeaders=[];for(var h=0;h<24;h++)colHeaders.push(h+'時');
  var colKeys=new Array(24).fill(0).map(function(_,i){return 'col_'+i;});
  var gT=hTots.reduce(function(s,c){return s+c.total;},0);
  var gG=hTots.reduce(function(s,c){return s+c.got;},0);
  var html='<table>'+_typeTableHead(colHeaders,colKeys,viewId,tabId,spec)+'<tbody>';
  html+=_typeTotalBlock(hTots,gT,gG);
  types.forEach(function(t){
    var td=typeData[t];
    html+=_typeRows(typeNums[t],_typeName(t),{total:td.total,got:td.got,missed:td.total-td.got},mat[t]);
  });
  return html+'</tbody></table>';
}

function buildTypeWeekdayTab(rows, viewId, tabId) {
  var WDAYS=['月','火','水','木','金','土','日','祝'];
  var types=[],typeSet={},typeNums={};
  rows.forEach(function(r){
    (r.ext_daily||[]).forEach(function(e){
      if(extFilterType&&e.type!==extFilterType)return;
      if(extFilterExt&&e.ext!==extFilterExt)return;
      if(!typeSet[e.type]){typeSet[e.type]=true;types.push(e.type);}
    });
  });
  if(!types.length) return '<p style="color:#aaa">データなし</p>';
  types.sort(function(a,b){return (+a||0)-(+b||0)||a.localeCompare(b);});
  types.forEach(function(t,i){typeNums[t]=i+1;});
  var mat={};
  types.forEach(function(t){mat[t]={};WDAYS.forEach(function(w){mat[t][w]={total:0,got:0};});});
  rows.forEach(function(r){
    var wd=r.weekday;
    if(!wd||WDAYS.indexOf(wd)<0)return;
    (r.ext_daily||[]).forEach(function(e){
      if(extFilterType&&e.type!==extFilterType)return;
      if(extFilterExt&&e.ext!==extFilterExt)return;
      if(!mat[e.type])return;
      mat[e.type][wd].total+=e.total;mat[e.type][wd].got+=e.got;
    });
  });
  var wBuckets={};WDAYS.forEach(function(w){wBuckets[w]=[];});
  rows.forEach(function(r){if(r.weekday&&wBuckets[r.weekday])wBuckets[r.weekday].push(r);});
  var typeData={};
  types.forEach(function(t){
    var rT=0,rG=0;
    var colV=WDAYS.map(function(w){var c=mat[t][w];rT+=c.total;rG+=c.got;return c.total;});
    typeData[t]={total:rT,got:rG,colV:colV};
  });
  var spec=_mTSort[viewId+'-'+tabId]||{key:'num',dir:'asc'};
  _sortTypes(types,spec,typeNums,typeData);
  var colAgg=WDAYS.map(function(w){
    var wrs=wBuckets[w];var wT=0,wG=0;
    wrs.forEach(function(r){var ft=_getFilteredTotals(r);wT+=ft.total;wG+=ft.got;});
    return{total:wT,got:wG};
  });
  var gT=colAgg.reduce(function(s,c){return s+c.total;},0),gG=colAgg.reduce(function(s,c){return s+c.got;},0);
  var colKeys=WDAYS.map(function(_,i){return 'col_'+i;});
  var html='<table>'+_typeTableHead(WDAYS,colKeys,viewId,tabId,spec)+'<tbody>';
  html+=_typeTotalBlock(colAgg,gT,gG);
  types.forEach(function(t){
    var td=typeData[t];
    var cols=WDAYS.map(function(w){return mat[t][w];});
    html+=_typeRows(typeNums[t],_typeName(t),{total:td.total,got:td.got,missed:td.total-td.got},cols);
  });
  return html+'</tbody></table>';
}

function buildDayHourTab(rows, viewId, tabId) {
  rows=rows.slice().sort(function(a,b){return a.date.localeCompare(b.date);});
  if(!rows.length) return '<p style="color:#aaa">データなし</p>';
  var rowData=rows.map(function(r){
    var ft=_getFilteredTotals(r);
    var hh=new Array(24).fill(null).map(function(_,h){return (r.hourly&&r.hourly[h])||{total:0,got:0};});
    return{r:r,ft:ft,hh:hh,colV:hh.map(function(c){return c.total;})};
  });
  var spec=_mTSort[viewId+'-'+tabId]||{key:'date',dir:'asc'};
  var dd=spec.dir==='desc'?-1:1;
  rowData.sort(function(a,b){
    var k=spec.key||'date';
    if(k==='date'||k==='weekday') return dd*a.r.date.localeCompare(b.r.date);
    if(k==='total') return dd*((a.ft.total||0)-(b.ft.total||0));
    var mi=k.match(/^col_([0-9]+)$/);
    if(mi){var ci=+mi[1];return dd*((a.colV[ci]||0)-(b.colV[ci]||0));}
    return 0;
  });
  var hTots=new Array(24).fill(null).map(function(){return{total:0,got:0};});
  rowData.forEach(function(rd){for(var h=0;h<24;h++){hTots[h].total+=rd.hh[h].total;hTots[h].got+=rd.hh[h].got;}});
  var gT=hTots.reduce(function(s,c){return s+c.total;},0),gG=hTots.reduce(function(s,c){return s+c.got;},0);
  var totW=rowData.reduce(function(s,rd){return s+(rd.ft.work_hrs||0);},0);
  var totTs=rowData.reduce(function(s,rd){return s+(rd.ft.avg_talk_secs&&rd.ft.got?rd.ft.avg_talk_secs*rd.ft.got:0);},0);
  var colLabels=[],colKeys=[];for(var h=0;h<24;h++){colLabels.push(h+'\u6642');colKeys.push('col_'+h);}
  var LABELS=['\u7740\u4fe1\u6570','\u5fdc\u7b54\u6570','\u672a\u5fdc\u7b54\u6570','\u5fdc\u7b54\u7387','\u7a3c\u50cd\u6642\u9593','CPH','\u901a\u8a71\u6642\u9593'];
  var html='<table>'+_dayHourHead(colLabels,colKeys,viewId,tabId,spec)+'<tbody>';
  LABELS.forEach(function(lbl,li){
    var cls=li===0?' class="total-block row-grp-first"':' class="total-block"';
    html+='<tr'+cls+'>';
    if(li===0){html+='<td class="sticky-l1 total-block" rowspan="7">\u5408\u8a08</td><td class="sticky-l2 total-block" rowspan="7"></td>';}
    html+='<td class="sticky-l3 metric-lbl">'+lbl+'</td>';
    hTots.forEach(function(c){
      if(li===0)html+='<td>'+c.total+'</td>';
      else if(li===1)html+='<td>'+c.got+'</td>';
      else if(li===2)html+='<td>'+(c.total-c.got)+'</td>';
      else if(li===3){if(!c.total)html+='<td style="color:#ccc">-</td>';else{var r=Math.round(c.got/c.total*1000)/10;html+='<td class="'+rateCls(r)+'">'+r+'%</td>';}}
      else html+='<td style="color:#ccc">-</td>';
    });
    if(li===0)html+='<td>'+gT+'</td>';
    else if(li===1)html+='<td>'+gG+'</td>';
    else if(li===2)html+='<td>'+(gT-gG)+'</td>';
    else if(li===3){if(!gT)html+='<td style="color:#ccc">-</td>';else{var r=Math.round(gG/gT*1000)/10;html+='<td class="'+rateCls(r)+'">'+r+'%</td>';}}
    else if(li===4)html+='<td>'+_fmtW(totW)+'</td>';
    else if(li===5){var cph=_cphVal(gG,totW);html+='<td class="'+cphCls(cph)+'">'+(cph!=null?cph:'-')+'</td>';}
    else html+='<td>'+_fmtTalk(gG>0?Math.round(totTs/gG):null)+'</td>';
    html+='</tr>';
  });
  rowData.forEach(function(rd){
    var r=rd.r,ft=rd.ft,hh=rd.hh,cph=_cphVal(ft.got,ft.work_hrs||0);
    LABELS.forEach(function(lbl,li){
      var cls=li===0?' class="row-grp-first"':'';
      html+='<tr'+cls+'>';
      if(li===0){html+='<td class="sticky-l1" rowspan="7">'+r.date.slice(5)+'</td><td class="sticky-l2" rowspan="7">'+(r.weekday||'')+'</td>';}
      html+='<td class="sticky-l3 metric-lbl">'+lbl+'</td>';
      hh.forEach(function(c){
        if(li===0)html+='<td>'+(c.total||0)+'</td>';
        else if(li===1)html+='<td>'+(c.got||0)+'</td>';
        else if(li===2)html+='<td>'+((c.total||0)-(c.got||0))+'</td>';
        else if(li===3){if(!c.total)html+='<td style="color:#ccc">-</td>';else{var rv=Math.round(c.got/c.total*1000)/10;html+='<td class="'+rateCls(rv)+'">'+rv+'%</td>';}}
        else html+='<td style="color:#ccc">-</td>';
      });
      if(li===0)html+='<td>'+ft.total+'</td>';
      else if(li===1)html+='<td>'+ft.got+'</td>';
      else if(li===2)html+='<td>'+ft.missed+'</td>';
      else if(li===3){if(!ft.total)html+='<td style="color:#ccc">-</td>';else{var rv=Math.round(ft.got/ft.total*1000)/10;html+='<td class="'+rateCls(rv)+'">'+rv+'%</td>';}}
      else if(li===4)html+='<td>'+_fmtW(ft.work_hrs||0)+'</td>';
      else if(li===5)html+='<td class="'+cphCls(cph)+'">'+(cph!=null?cph:'-')+'</td>';
      else html+='<td>'+_fmtTalk(ft.avg_talk_secs)+'</td>';
      html+='</tr>';
    });
  });
  return html+'</tbody></table>';
}

function buildWeekdayHourTab(rows, viewId, tabId) {
  var WDAYS=['\u6708','\u706b','\u6c34','\u6728','\u91d1','\u571f','\u65e5','\u795d'];
  var wBuckets={};WDAYS.forEach(function(w){wBuckets[w]=[];});
  rows.forEach(function(r){if(r.weekday&&wBuckets[r.weekday])wBuckets[r.weekday].push(r);});
  var wHMat={};
  WDAYS.forEach(function(w){wHMat[w]=new Array(24).fill(null).map(function(){return{total:0,got:0};});});
  rows.forEach(function(r){
    var wd=r.weekday;if(!wd||!wHMat[wd])return;
    for(var h=0;h<24;h++){var hh=r.hourly&&r.hourly[h];if(hh){wHMat[wd][h].total+=hh.total||0;wHMat[wd][h].got+=hh.got||0;}}
  });
  var wdData={};
  WDAYS.forEach(function(w){
    var wrs=wBuckets[w];
    var wT=wHMat[w].reduce(function(s,c){return s+c.total;},0);
    var wG=wHMat[w].reduce(function(s,c){return s+c.got;},0);
    var wW=wrs.reduce(function(s,r){return s+(r.work_hrs||0);},0);
    wdData[w]={total:wT,got:wG,work_hrs:wW,cnt:wrs.length,colV:wHMat[w].map(function(c){return c.total;})};
  });
  var activeWdays=WDAYS.filter(function(w){return wBuckets[w].length>0;});
  var spec=_mTSort[viewId+'-'+tabId]||{key:'weekday',dir:'asc'};
  var dd=spec.dir==='desc'?-1:1;
  activeWdays.sort(function(a,b){
    var k=spec.key||'weekday';
    if(k==='weekday'||k==='cnt') return dd*(WDAYS.indexOf(a)-WDAYS.indexOf(b));
    if(k==='total') return dd*((wdData[a].total||0)-(wdData[b].total||0));
    var mi=k.match(/^col_([0-9]+)$/);
    if(mi){var ci=+mi[1];return dd*((wdData[a].colV[ci]||0)-(wdData[b].colV[ci]||0));}
    return 0;
  });
  var hTots=new Array(24).fill(null).map(function(){return{total:0,got:0};});
  activeWdays.forEach(function(w){for(var h=0;h<24;h++){hTots[h].total+=wHMat[w][h].total;hTots[h].got+=wHMat[w][h].got;}});
  var gT=hTots.reduce(function(s,c){return s+c.total;},0),gG=hTots.reduce(function(s,c){return s+c.got;},0);
  var totW=activeWdays.reduce(function(s,w){return s+wdData[w].work_hrs;},0);
  var colLabels=[],colKeys=[];for(var h=0;h<24;h++){colLabels.push(h+'\u6642');colKeys.push('col_'+h);}
  var LABELS=['\u7740\u4fe1\u6570','\u5fdc\u7b54\u6570','\u672a\u5fdc\u7b54\u6570','\u5fdc\u7b54\u7387','\u7a3c\u50cd\u6642\u9593','CPH'];
  var html='<table>'+_wdHourHead(colLabels,colKeys,viewId,tabId,spec)+'<tbody>';
  LABELS.forEach(function(lbl,li){
    var cls=li===0?' class="total-block row-grp-first"':' class="total-block"';
    html+='<tr'+cls+'>';
    if(li===0){html+='<td class="sticky-l1 total-block" rowspan="6">\u5408\u8a08</td><td class="sticky-l2 total-block" rowspan="6">'+rows.length+'\u65e5</td>';}
    html+='<td class="sticky-l3 metric-lbl">'+lbl+'</td>';
    hTots.forEach(function(c){
      if(li===0)html+='<td>'+c.total+'</td>';
      else if(li===1)html+='<td>'+c.got+'</td>';
      else if(li===2)html+='<td>'+(c.total-c.got)+'</td>';
      else if(li===3){if(!c.total)html+='<td style="color:#ccc">-</td>';else{var r=Math.round(c.got/c.total*1000)/10;html+='<td class="'+rateCls(r)+'">'+r+'%</td>';}}
      else html+='<td style="color:#ccc">-</td>';
    });
    if(li===0)html+='<td>'+gT+'</td>';
    else if(li===1)html+='<td>'+gG+'</td>';
    else if(li===2)html+='<td>'+(gT-gG)+'</td>';
    else if(li===3){if(!gT)html+='<td style="color:#ccc">-</td>';else{var r=Math.round(gG/gT*1000)/10;html+='<td class="'+rateCls(r)+'">'+r+'%</td>';}}
    else if(li===4)html+='<td>'+_fmtW(totW)+'</td>';
    else{var cph=_cphVal(gG,totW);html+='<td class="'+cphCls(cph)+'">'+(cph!=null?cph:'-')+'</td>';}
    html+='</tr>';
  });
  activeWdays.forEach(function(w){
    var wd=wdData[w],cph=_cphVal(wd.got,wd.work_hrs);
    LABELS.forEach(function(lbl,li){
      var cls=li===0?' class="row-grp-first"':'';
      html+='<tr'+cls+'>';
      if(li===0){html+='<td class="sticky-l1" rowspan="6">'+w+'</td><td class="sticky-l2" rowspan="6">'+wd.cnt+'\u65e5</td>';}
      html+='<td class="sticky-l3 metric-lbl">'+lbl+'</td>';
      wHMat[w].forEach(function(c){
        if(li===0)html+='<td>'+c.total+'</td>';
        else if(li===1)html+='<td>'+c.got+'</td>';
        else if(li===2)html+='<td>'+(c.total-c.got)+'</td>';
        else if(li===3){if(!c.total)html+='<td style="color:#ccc">-</td>';else{var r=Math.round(c.got/c.total*1000)/10;html+='<td class="'+rateCls(r)+'">'+r+'%</td>';}}
        else html+='<td style="color:#ccc">-</td>';
      });
      if(li===0)html+='<td>'+wd.total+'</td>';
      else if(li===1)html+='<td>'+wd.got+'</td>';
      else if(li===2)html+='<td>'+(wd.total-wd.got)+'</td>';
      else if(li===3){if(!wd.total)html+='<td style="color:#ccc">-</td>';else{var r=Math.round(wd.got/wd.total*1000)/10;html+='<td class="'+rateCls(r)+'">'+r+'%</td>';}}
      else if(li===4)html+='<td>'+_fmtW(wd.work_hrs)+'</td>';
      else html+='<td class="'+cphCls(cph)+'">'+(cph!=null?cph:'-')+'</td>';
      html+='</tr>';
    });
  });
  return html+'</tbody></table>';
}

function buildOpDayTab(rows, viewId, tabId) {
  var ops=[],opSet={},opNames={};
  rows.forEach(function(r){
    (r.op_daily||[]).forEach(function(o){
      if(!opSet[o.zoiper]){opSet[o.zoiper]=true;ops.push(o.zoiper);opNames[o.zoiper]=o.name;}
    });
  });
  if(!ops.length) return '<p style="color:#aaa">Zoiper\u30c7\u30fc\u30bf\u306a\u3057\uff08\u518d\u5206\u6790\u5f8c\u306b\u8868\u793a\u3055\u308c\u307e\u3059\uff09</p>';
  var mat={};
  ops.forEach(function(z){mat[z]={};});
  rows.forEach(function(r){
    (r.op_daily||[]).forEach(function(o){
      if(!mat[o.zoiper])return;
      mat[o.zoiper][r.date]={total:o.total,got:o.got,missed:o.missed,work_hrs:o.work_hrs||0,talk_sum:o.talk_sum||0};
    });
  });
  var opData={};
  ops.forEach(function(z){
    var totT=0,totG=0,totW=0,totTs=0;
    var colV=rows.map(function(r){var c=mat[z][r.date];if(c){totT+=c.total;totG+=c.got;totW+=c.work_hrs||0;totTs+=c.talk_sum||0;}return c?c.got:0;});
    opData[z]={total:totT,got:totG,work_hrs:totW,avg_talk_secs:totG>0?Math.round(totTs/totG):null,colV:colV};
  });
  var spec=_mTSort[viewId+'-'+tabId]||_mTSortDef[tabId]||{key:'cph',dir:'desc'};
  _sortOps(ops,spec,opData,opNames);
  var colLabels=rows.map(function(r){return r.date.slice(5)+'<br>'+(r.weekday||'');});
  var colKeys=rows.map(function(_,i){return 'col_'+i;});
  var colAgg=rows.map(function(r){return{total:r.total,got:r.got};});
  var gT=rows.reduce(function(s,r){return s+r.total;},0),gG=rows.reduce(function(s,r){return s+r.got;},0);
  var gW=rows.reduce(function(s,r){return s+(r.work_hrs||0);},0);
  var gTs=rows.reduce(function(s,r){return s+(r.avg_talk_secs&&r.got?r.avg_talk_secs*r.got:0);},0);
  var grandTotals={total:gT,got:gG,missed:gT-gG,work_hrs:gW,avg_talk_secs:gG>0?Math.round(gTs/gG):null};
  var html='<table>'+_opTableHead(colLabels,colKeys,viewId,tabId,spec)+'<tbody>';
  html+=_opTotalBlock(colAgg,grandTotals);
  ops.forEach(function(z){
    var od=opData[z];
    var cols=rows.map(function(r){return mat[z][r.date]||{total:0,got:0};});
    html+=_opRows(z,opNames[z]||z,{total:od.total,got:od.got,missed:od.total-od.got,work_hrs:od.work_hrs,avg_talk_secs:od.avg_talk_secs},cols);
  });
  return html+'</tbody></table>';
}

function buildOpHourTab(rows, viewId, tabId) {
  var ops=[],opSet={},opNames={};
  rows.forEach(function(r){
    (r.op_daily||[]).forEach(function(o){
      if(!opSet[o.zoiper]){opSet[o.zoiper]=true;ops.push(o.zoiper);opNames[o.zoiper]=o.name;}
    });
  });
  if(!ops.length) return '<p style="color:#aaa">Zoiper\u30c7\u30fc\u30bf\u306a\u3057\uff08\u518d\u5206\u6790\u5f8c\u306b\u8868\u793a\u3055\u308c\u307e\u3059\uff09</p>';
  var mat={};
  ops.forEach(function(z){mat[z]=new Array(24).fill(0);});
  rows.forEach(function(r){
    (r.op_daily||[]).forEach(function(o){
      if(!mat[o.zoiper])return;
      for(var h=0;h<24;h++)mat[o.zoiper][h]+=(o.h_got||[])[h]||0;
    });
  });
  var hTots=new Array(24).fill(null).map(function(){return{total:0,got:0};});
  rows.forEach(function(r){for(var h=0;h<24;h++){var hh=r.hourly&&r.hourly[h];if(hh){hTots[h].total+=hh.total||0;hTots[h].got+=hh.got||0;}}});
  var opData={};
  ops.forEach(function(z){
    var totT=0,totG=0,totW=0,totTs=0;
    rows.forEach(function(r){var o=(r.op_daily||[]).find(function(x){return x.zoiper===z;});if(o){totT+=o.total;totG+=o.got;totW+=o.work_hrs||0;totTs+=o.talk_sum||0;}});
    opData[z]={total:totT,got:totG,work_hrs:totW,avg_talk_secs:totG>0?Math.round(totTs/totG):null,colV:mat[z].slice()};
  });
  var spec=_mTSort[viewId+'-'+tabId]||_mTSortDef[tabId]||{key:'cph',dir:'desc'};
  _sortOps(ops,spec,opData,opNames);
  var gT=hTots.reduce(function(s,c){return s+c.total;},0),gG=hTots.reduce(function(s,c){return s+c.got;},0);
  var gW=rows.reduce(function(s,r){return s+(r.work_hrs||0);},0);
  var gTs=rows.reduce(function(s,r){return s+(r.avg_talk_secs&&r.got?r.avg_talk_secs*r.got:0);},0);
  var grandTotals={total:gT,got:gG,missed:gT-gG,work_hrs:gW,avg_talk_secs:gG>0?Math.round(gTs/gG):null};
  var colLabels=[],colKeys=[];for(var h=0;h<24;h++){colLabels.push(h+'\u6642');colKeys.push('col_'+h);}
  var html='<table>'+_opTableHead(colLabels,colKeys,viewId,tabId,spec)+'<tbody>';
  html+=_opTotalBlock(hTots,grandTotals);
  ops.forEach(function(z){
    var od=opData[z];
    var cols=mat[z].map(function(g){return{total:null,got:g};});
    html+=_opRows(z,opNames[z]||z,{total:od.total,got:od.got,missed:od.total-od.got,work_hrs:od.work_hrs,avg_talk_secs:od.avg_talk_secs},cols);
  });
  return html+'</tbody></table>';
}

function buildOpWeekdayTab(rows, viewId, tabId) {
  var WDAYS=['\u6708','\u706b','\u6c34','\u6728','\u91d1','\u571f','\u65e5','\u795d'];
  var ops=[],opSet={},opNames={};
  rows.forEach(function(r){
    (r.op_daily||[]).forEach(function(o){
      if(!opSet[o.zoiper]){opSet[o.zoiper]=true;ops.push(o.zoiper);opNames[o.zoiper]=o.name;}
    });
  });
  if(!ops.length) return '<p style="color:#aaa">Zoiper\u30c7\u30fc\u30bf\u306a\u3057\uff08\u518d\u5206\u6790\u5f8c\u306b\u8868\u793a\u3055\u308c\u307e\u3059\uff09</p>';
  var mat={};
  ops.forEach(function(z){mat[z]={};WDAYS.forEach(function(w){mat[z][w]={total:0,got:0,work_hrs:0,talk_sum:0};});});
  rows.forEach(function(r){
    var wd=r.weekday;if(!wd||WDAYS.indexOf(wd)<0)return;
    (r.op_daily||[]).forEach(function(o){
      if(!mat[o.zoiper])return;
      mat[o.zoiper][wd].total+=o.total;mat[o.zoiper][wd].got+=o.got;
      mat[o.zoiper][wd].work_hrs+=o.work_hrs||0;mat[o.zoiper][wd].talk_sum+=o.talk_sum||0;
    });
  });
  var wBuckets={};WDAYS.forEach(function(w){wBuckets[w]=[];});
  rows.forEach(function(r){if(r.weekday&&wBuckets[r.weekday])wBuckets[r.weekday].push(r);});
  var opData={};
  ops.forEach(function(z){
    var totT=0,totG=0,totW=0,totTs=0;
    var colV=WDAYS.map(function(w){var c=mat[z][w];totT+=c.total;totG+=c.got;totW+=c.work_hrs;totTs+=c.talk_sum;return c.got;});
    opData[z]={total:totT,got:totG,work_hrs:totW,avg_talk_secs:totG>0?Math.round(totTs/totG):null,colV:colV};
  });
  var spec=_mTSort[viewId+'-'+tabId]||_mTSortDef[tabId]||{key:'cph',dir:'desc'};
  _sortOps(ops,spec,opData,opNames);
  var colAgg=WDAYS.map(function(w){
    var wrs=wBuckets[w];return{total:wrs.reduce(function(s,r){return s+r.total;},0),got:wrs.reduce(function(s,r){return s+r.got;},0)};
  });
  var gT=rows.reduce(function(s,r){return s+r.total;},0),gG=rows.reduce(function(s,r){return s+r.got;},0);
  var gW=rows.reduce(function(s,r){return s+(r.work_hrs||0);},0);
  var gTs=rows.reduce(function(s,r){return s+(r.avg_talk_secs&&r.got?r.avg_talk_secs*r.got:0);},0);
  var grandTotals={total:gT,got:gG,missed:gT-gG,work_hrs:gW,avg_talk_secs:gG>0?Math.round(gTs/gG):null};
  var colKeys=WDAYS.map(function(_,i){return 'col_'+i;});
  var html='<table>'+_opTableHead(WDAYS,colKeys,viewId,tabId,spec)+'<tbody>';
  html+=_opTotalBlock(colAgg,grandTotals);
  ops.forEach(function(z){
    var od=opData[z];
    var cols=WDAYS.map(function(w){return mat[z][w];});
    html+=_opRows(z,opNames[z]||z,{total:od.total,got:od.got,missed:od.total-od.got,work_hrs:od.work_hrs,avg_talk_secs:od.avg_talk_secs},cols);
  });
  return html+'</tbody></table>';
}

function buildOpTypeTab(rows, viewId, tabId) {
  var ops=[],opSet={},opNames={};
  var types=[],typeSet={},typeNums={};
  rows.forEach(function(r){
    (r.op_daily||[]).forEach(function(o){
      if(!opSet[o.zoiper]){opSet[o.zoiper]=true;ops.push(o.zoiper);opNames[o.zoiper]=o.name;}
      Object.keys(o.type_total||{}).forEach(function(t){if(!typeSet[t]){typeSet[t]=true;types.push(t);}});
      Object.keys(o.type_got||{}).forEach(function(t){if(!typeSet[t]){typeSet[t]=true;types.push(t);}});
    });
    (r.ext_daily||[]).forEach(function(e){
      if(!typeSet[e.type]){typeSet[e.type]=true;types.push(e.type);}
    });
  });
  if(!ops.length||!types.length) return '<p style="color:#aaa">\u30c7\u30fc\u30bf\u306a\u3057</p>';
  types.sort(function(a,b){return (+a||0)-(+b||0)||a.localeCompare(b);});
  types.forEach(function(t,i){typeNums[t]=i+1;});
  var mat={};
  ops.forEach(function(z){mat[z]={};types.forEach(function(t){mat[z][t]={total:0,got:0};});});
  rows.forEach(function(r){
    (r.op_daily||[]).forEach(function(o){
      if(!mat[o.zoiper])return;
      var tg=o.type_got||{},tt=o.type_total||{};
      types.forEach(function(t){mat[o.zoiper][t].got+=(tg[t]||0);mat[o.zoiper][t].total+=(tt[t]||tg[t]||0);});
    });
  });
  var opData={};
  ops.forEach(function(z){
    var totT=0,totG=0,totW=0,totTs=0;
    rows.forEach(function(r){var o=(r.op_daily||[]).find(function(x){return x.zoiper===z;});if(o){totT+=o.total;totG+=o.got;totW+=o.work_hrs||0;totTs+=o.talk_sum||0;}});
    opData[z]={total:totT,got:totG,work_hrs:totW,avg_talk_secs:totG>0?Math.round(totTs/totG):null,colV:types.map(function(t){return mat[z][t].got;})};
  });
  var spec=_mTSort[viewId+'-'+tabId]||_mTSortDef[tabId]||{key:'cph',dir:'desc'};
  _sortOps(ops,spec,opData,opNames);
  var colLabels=types.map(function(t){return typeNums[t]+'<br><small>'+_typeName(t)+'</small>';});
  var colKeys=types.map(function(_,i){return 'col_'+i;});
  var colAgg=types.map(function(t){
    var tot=ops.reduce(function(s,z){return s+mat[z][t].total;},0);
    var got=ops.reduce(function(s,z){return s+mat[z][t].got;},0);
    return{total:tot,got:got};
  });
  var gT=rows.reduce(function(s,r){return s+r.total;},0),gG=rows.reduce(function(s,r){return s+r.got;},0);
  var gW=rows.reduce(function(s,r){return s+(r.work_hrs||0);},0);
  var gTs=rows.reduce(function(s,r){return s+(r.avg_talk_secs&&r.got?r.avg_talk_secs*r.got:0);},0);
  var grandTotals={total:gT,got:gG,missed:gT-gG,work_hrs:gW,avg_talk_secs:gG>0?Math.round(gTs/gG):null};
  var html='<table>'+_opTableHead(colLabels,colKeys,viewId,tabId,spec)+'<tbody>';
  html+=_opTotalBlock(colAgg,grandTotals);
  ops.forEach(function(z){
    var od=opData[z];
    var cols=types.map(function(t){return mat[z][t];});
    html+=_opRows(z,opNames[z]||z,{total:od.total,got:od.got,missed:od.total-od.got,work_hrs:od.work_hrs,avg_talk_secs:od.avg_talk_secs},cols);
  });
  return html+'</tbody></table>';
}


function renderWeekdayView(el) {
  var rows = getFilteredDays();
  if (!rows.length) {
    el.innerHTML = '<h2>曜日別累計</h2><p style="color:#aaa">データなし</p>';
    return;
  }
  var allDates = rows.map(function(r) { return r.date; }).sort();
  var period = allDates[0] + ' 〜 ' + allDates[allDates.length - 1];
  var days = ['月','火','水','木','金','土','日','祝'];
  var buckets = {};
  days.forEach(function(d) { buckets[d] = []; });
  rows.forEach(function(r) { if (r.weekday && buckets[r.weekday] !== undefined) buckets[r.weekday].push(r); });
  var fl = extFilterLabel();
  var titleSuffix = fl ? ' <span style="font-size:0.75em;background:#e8f0fe;color:#1a73e8;border-radius:3px;padding:2px 6px">' + fl + '</span>' : '';
  var html = '<h2>曜日別累計' + titleSuffix + '</h2>' +
    '<p style="color:#777;font-size:0.9em">対象期間: ' + period + '（' + rows.length + '日）</p>';
  html += '<table><tr><th>曜日</th><th>日数</th><th>着信合計</th><th>平均着信</th><th>平均応答</th><th>平均未応答</th><th>平均応答率</th><th>平均CPH</th><th>平均通話時間</th></tr>';
  days.forEach(function(day) {
    var rs = buckets[day];
    if (!rs.length) {
      html += '<tr><td>' + day + '</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>';
      return;
    }
    var n = rs.length;
    var sumT = rs.reduce(function(s,r){return s+r.total;},0);
    var sumG = rs.reduce(function(s,r){return s+r.got;},0);
    var sumM = rs.reduce(function(s,r){return s+r.missed;},0);
    var rate = sumT > 0 ? Math.round(sumG/sumT*1000)/10 : null;
    var cArr = rs.filter(function(r){return r.avg_cph != null;});
    var avgCph = cArr.length ? Math.round(cArr.reduce(function(s,r){return s+r.avg_cph;},0)/cArr.length*10)/10 : null;
    var tArr = rs.filter(function(r){return r.avg_talk_secs != null;});
    var avgTalk = tArr.length ? Math.round(tArr.reduce(function(s,r){return s+r.avg_talk_secs;},0)/tArr.length) : null;
    html += '<tr><td>' + day + '</td><td>' + n + '</td>' +
      '<td>' + sumT + '</td>' +
      '<td>' + Math.round(sumT/n*10)/10 + '</td><td>' + Math.round(sumG/n*10)/10 + '</td><td>' + Math.round(sumM/n*10)/10 + '</td>' +
      '<td class="' + rateCls(rate) + '">' + (rate != null ? rate + '%' : '-') + '</td>' +
      '<td class="' + cphCls(avgCph) + '">' + (avgCph != null ? avgCph : '-') + '</td>' +
      '<td>' + fmtSecs(avgTalk) + '</td></tr>';
  });
  html += '</table>';
  days.forEach(function(day) {
    var rs = buckets[day];
    if (!rs.length) return;
    var n = rs.length;
    html += '<h3>' + day + '曜日 時間帯別累計</h3>';
    html += '<table><tr><th>時間帯</th><th>着信合計</th><th>平均着信</th><th>応答合計</th><th>未応答合計</th><th>応答率</th><th>平均人員</th><th>平均通話時間</th></tr>';
    for (var h = 0; h < 24; h++) {
      var hT=0, hG=0, hM=0, hOps=0, hTalkS=0, hTalkN=0, hDays=0;
      rs.forEach(function(r) {
        if (!r.hourly || !r.hourly[h]) return;
        var hh = r.hourly[h];
        hT += hh.total||0; hG += hh.got||0; hM += hh.missed||0; hOps += hh.ops||0;
        if (hh.avg_talk != null) { hTalkS += hh.avg_talk; hTalkN++; }
        if ((hh.total||0) > 0) hDays++;
      });
      var hRate = hT > 0 ? Math.round(hG/hT*1000)/10 : null;
      html += '<tr><td>' + h + '時台</td>' +
        '<td>' + hT + '</td>' +
        '<td>' + (hDays > 0 ? Math.round(hT/hDays*10)/10 : '-') + '</td>' +
        '<td>' + hG + '</td><td>' + hM + '</td>' +
        '<td class="' + rateCls(hRate) + '">' + (hRate != null ? hRate + '%' : '-') + '</td>' +
        '<td>' + Math.round(hOps/n*10)/10 + '</td>' +
        '<td>' + fmtSecs(hTalkN > 0 ? Math.round(hTalkS/hTalkN) : null) + '</td></tr>';
    }
    html += '</table>';
  });
  el.innerHTML = html;
}
function renderHourView(el) {
  var rows = getFilteredDays();
  if (!rows.length) {
    el.innerHTML = '<h2>時間帯別累計</h2><p style="color:#aaa">データなし</p>';
    return;
  }
  var allDates = rows.map(function(r) { return r.date; }).sort();
  var period = allDates[0] + ' 〜 ' + allDates[allDates.length - 1];
  var N = rows.length;
  var hrs = [];
  for (var h = 0; h < 24; h++) {
    var sumT=0,sumG=0,sumM=0,opsSum=0,talkSum=0,talkN=0,hDays=0;
    rows.forEach(function(r) {
      if (!r.hourly || !r.hourly[h]) return;
      var hh = r.hourly[h];
      sumT += hh.total||0; sumG += hh.got||0; sumM += hh.missed||0; opsSum += hh.ops||0;
      if (hh.avg_talk != null) { talkSum += hh.avg_talk; talkN++; }
      if ((hh.total||0) > 0) hDays++;
    });
    hrs.push({h:h, total:sumT, got:sumG, missed:sumM, hDays:hDays,
      rate: sumT>0 ? Math.round(sumG/sumT*1000)/10 : null,
      avgOps: Math.round(opsSum/N*10)/10,
      avgTalk: talkN>0 ? Math.round(talkSum/talkN) : null});
  }
  var fl2 = extFilterLabel();
  var ts2 = fl2 ? ' <span style="font-size:0.75em;background:#e8f0fe;color:#1a73e8;border-radius:3px;padding:2px 6px">' + fl2 + '</span>' : '';
  var html = '<h2>時間帯別累計' + ts2 + '</h2>' +
    '<p style="color:#777;font-size:0.9em">対象期間: ' + period + '（全' + N + '日分）</p><table>' +
    '<tr><th>時間帯</th><th>着信合計</th><th>平均着信</th><th>応答合計</th><th>未応答合計</th><th>応答率</th><th>平均人員数</th><th>平均通話時間</th></tr>';
  hrs.forEach(function(r) {
    var rc = rateCls(r.rate);
    html += '<tr><td>' + r.h + '時台</td>' +
      '<td>' + r.total + '</td>' +
      '<td>' + (r.hDays > 0 ? Math.round(r.total/r.hDays*10)/10 : '-') + '</td>' +
      '<td>' + r.got + '</td><td>' + r.missed + '</td>' +
      '<td class="' + rc + '">' + (r.rate != null ? r.rate + '%' : '-') + '</td>' +
      '<td>' + r.avgOps + '</td>' +
      '<td>' + fmtSecs(r.avgTalk) + '</td></tr>';
  });
  html += '</table>';
  el.innerHTML = html;
}
"""


def _scope_report_html(body_html: str, dk: str) -> str:
    for old, new in [
        ('id="detail-',                        f'id="detail-{dk}-'),
        ("id='detail-",                        f"id='detail-{dk}-"),
        ('id="arrow-',                         f'id="arrow-{dk}-'),
        ("id='arrow-",                         f"id='arrow-{dk}-"),
        ('onclick="toggleDetail(',             f'onclick="toggleDetail_{dk}('),
        ('id="search-input"',                  f'id="search-input-{dk}"'),
        ('for="search-input"',                 f'for="search-input-{dk}"'),
        ('oninput="searchFilter(this.value)"', f'oninput="searchFilter_{dk}(this.value)"'),
        ('id="search-count"',                  f'id="search-count-{dk}"'),
        ('id="ranking-section"',               f'id="ranking-section-{dk}"'),
        ('function toggleDetail(',             f'function toggleDetail_{dk}('),
        ('function searchFilter(',             f'function searchFilter_{dk}('),
        ("getElementById('detail-' + h)",      f"getElementById('detail-{dk}-' + h)"),
        ("getElementById('arrow-' + h)",       f"getElementById('arrow-{dk}-' + h)"),
        ("getElementById('ranking-section')",  f"getElementById('ranking-section-{dk}')"),
        ("getElementById('search-count')",     f"getElementById('search-count-{dk}')"),
    ]:
        body_html = body_html.replace(old, new)
    return body_html


def build_dashboard() -> None:
    today  = date.today()
    _cm = today.month - 2
    _cy = today.year + (_cm - 1) // 12
    _cm = (_cm - 1) % 12 + 1
    cutoff = date(_cy, _cm, 1)

    reports: list[tuple[date, Path]] = []
    for p in sorted(DAILY_DIR.glob("*.html")):
        try:
            d = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if d < cutoff:
            continue
        reports.append((d, p))

    if not reports:
        print("⚠️  ダッシュボード: 対象レポートなし（スキップ）")
        return

    available_dates = [str(d) for d, _ in reports]
    latest          = reports[-1][0]

    sections_html  = ""
    all_daily_data: list[dict] = []
    for d, p in reports:
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError as e:
            print(f"⚠️  ダッシュボード: {p.name} 読み込み失敗 ({e})", file=sys.stderr)
            continue
        m = re.search(r"<body>(.*?)</body>", raw, re.DOTALL)
        if not m:
            print(f"⚠️  ダッシュボード: {p.name} body抽出失敗（スキップ）", file=sys.stderr)
            continue
        # extract embedded JSON summary; fall back to MD sidecar for older HTML files
        m2 = re.search(r'<script type="application/json" id="rdata">(.*?)</script>',
                       raw, re.DOTALL)
        if m2:
            try:
                all_daily_data.append(json.loads(m2.group(1)))
            except (json.JSONDecodeError, ValueError):
                pass
        dk     = str(d).replace("-", "")
        scoped = _scope_report_html(m.group(1).strip(), dk)
        sections_html += (
            f'<div id="report-{d}" class="report-section">\n'
            f'{scoped}\n</div>\n'
        )

    dates_json      = str(available_dates).replace("'", '"')
    daily_data_json = json.dumps(all_daily_data, ensure_ascii=False)

    ext_type_path = DATA_DIR / "ext_type.csv"
    type_names: dict[str, str] = {}
    if ext_type_path.exists():
        try:
            _et = pd.read_csv(ext_type_path, encoding="utf-8-sig")
            for _, _row in _et.iterrows():
                _no = _row.get("種別名称一覧 No.")
                _nm = _row.get("名称")
                if pd.notna(_no) and pd.notna(_nm) and str(_no).strip():
                    type_names[str(int(float(_no)))] = str(_nm)
        except Exception:
            pass
    type_names_json = json.dumps(type_names, ensure_ascii=False)

    dashboard_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>コールセンター分析ダッシュボード</title>
{_CSS}
{_DASHBOARD_CSS}
</head>
<body>
<div id="dashboard-wrapper">
<div id="sidebar">
  <button id="sidebar-toggle" onclick="toggleSidebar()" title="サイドバーを折りたたむ">◀</button>
  <div id="sidebar-title">📊 分析ダッシュボード</div>
  <div id="gs-wrap">
    <input id="gs-input" type="text" placeholder="案件名・内線・OP名..." oninput="globalSearch(this.value)">
  </div>
  <div class="cal-nav">
    <button onclick="prevMonth()">◀</button>
    <span id="month-label"></span>
    <button onclick="nextMonth()">▶</button>
  </div>
  <div id="calendar-grid"></div>
  <div class="nav-group">
    <div class="nav-label">集計ビュー</div>
    <button class="nav-btn" data-view="prev-month"  onclick="showView('prev-month')">📅 前月累計</button>
    <button class="nav-btn" data-view="this-month"  onclick="showView('this-month')">📊 当月累計</button>
    <button class="nav-btn" data-view="by-weekday"  onclick="showView('by-weekday')">📆 曜日別累計</button>
    <button class="nav-btn" data-view="by-hour"     onclick="showView('by-hour')">⏰ 時間帯別累計</button>
  </div>
  <div id="legend-wrap">
    <div class="legend-title">凡例</div>
    <div class="legend-cat">応答率</div>
    <div class="legend-row"><span class="legend-swatch" style="background:#d4edda"></span><span class="legend-text">90%以上</span></div>
    <div class="legend-row"><span class="legend-swatch" style="background:#fff3cd"></span><span class="legend-text">80〜90%未満</span></div>
    <div class="legend-row"><span class="legend-swatch" style="background:#f8d7da"></span><span class="legend-text">80%未満</span></div>
    <div class="legend-cat">CPH</div>
    <div class="legend-row"><span class="legend-swatch" style="background:#d4edda"></span><span class="legend-text">10以上</span></div>
    <div class="legend-row"><span class="legend-swatch" style="background:#fff3cd"></span><span class="legend-text">8〜10未満</span></div>
    <div class="legend-row"><span class="legend-swatch" style="background:#f8d7da"></span><span class="legend-text">8未満</span></div>
    <div class="legend-cat">KPI評価（案件別）</div>
    <div class="legend-row"><span class="kpi-badge kpi-s">S</span><span class="legend-text" style="margin-left:4px">98%以上</span></div>
    <div class="legend-row"><span class="kpi-badge kpi-a">A</span><span class="legend-text" style="margin-left:4px">95〜98%未満</span></div>
    <div class="legend-row"><span class="kpi-badge kpi-b">B</span><span class="legend-text" style="margin-left:4px">90〜95%未満</span></div>
    <div class="legend-row"><span class="kpi-badge kpi-c">C</span><span class="legend-text" style="margin-left:4px">85〜90%未満</span></div>
    <div class="legend-row"><span class="kpi-badge kpi-d">D</span><span class="legend-text" style="margin-left:4px">85%未満</span></div>
  </div>
</div>
<div id="main-content">
  <div id="view-filter-bar" style="display:none;background:#f8f8f8;border:1px solid #ddd;border-radius:6px;padding:8px 14px;margin-bottom:1.2em;align-items:center;gap:10px;flex-wrap:wrap">
    <label style="font-weight:bold;color:#444;white-space:nowrap">期間：</label>
    <input type="date" id="filter-from" onchange="applyViewFilter()" style="padding:4px 8px;border:1px solid #aaa;border-radius:4px">
    <span>〜</span>
    <input type="date" id="filter-to" onchange="applyViewFilter()" style="padding:4px 8px;border:1px solid #aaa;border-radius:4px">
    <span style="border-left:1px solid #ccc;align-self:stretch;margin:0 2px"></span>
    <label style="font-weight:bold;color:#444;white-space:nowrap">案件：</label>
    <select id="ext-type-filter" onchange="applyExtFilter()" style="padding:4px 8px;border:1px solid #aaa;border-radius:4px;font-size:0.9em"></select>
    <select id="ext-num-filter"  onchange="applyExtFilter()" style="padding:4px 8px;border:1px solid #aaa;border-radius:4px;font-size:0.9em;max-width:260px"></select>
    <button onclick="clearViewFilter()" style="padding:4px 10px;border:1px solid #aaa;border-radius:4px;background:#fff;cursor:pointer">リセット</button>
    <span id="filter-count" style="font-size:0.9em;color:#666"></span>
  </div>
  <div id="no-selection">← 日付を選択してください</div>
  <div id="search-results-panel"></div>
  <div id="view-prev-month"  class="view-section"></div>
  <div id="view-this-month"  class="view-section"></div>
  <div id="view-by-weekday"  class="view-section"></div>
  <div id="view-by-hour"     class="view-section"></div>
{sections_html}
</div>
</div>
<script>
const availableDates = {dates_json};
const dailyData = {daily_data_json};
const typeNames = {type_names_json};
let currentYear  = {latest.year};
let currentMonth = {latest.month - 1};
let selectedDate = null;

{_VIEWS_JS}

function renderCalendar() {{
  document.getElementById('month-label').textContent =
    currentYear + '年' + (currentMonth + 1) + '月';
  const grid = document.getElementById('calendar-grid');
  grid.innerHTML = '';
  ['日','月','火','水','木','金','土'].forEach(function(h) {{
    const th = document.createElement('div');
    th.className = 'cal-header';
    th.textContent = h;
    grid.appendChild(th);
  }});
  const firstDay    = new Date(currentYear, currentMonth, 1).getDay();
  const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
  const todayStr    = new Date().toLocaleDateString('sv');
  for (let i = 0; i < firstDay; i++) {{
    const el = document.createElement('div');
    el.className = 'cal-cell';
    grid.appendChild(el);
  }}
  for (let d = 1; d <= daysInMonth; d++) {{
    const ds = currentYear + '-' +
               String(currentMonth + 1).padStart(2, '0') + '-' +
               String(d).padStart(2, '0');
    const el = document.createElement('div');
    el.textContent = d;
    el.className   = 'cal-cell';
    if (availableDates.includes(ds)) {{
      el.classList.add('cal-available');
      (function(date) {{ el.onclick = function() {{ selectDate(date); }}; }})(ds);
    }}
    if (ds === selectedDate) {{ el.classList.add('cal-selected'); }}
    if (ds === todayStr)     {{ el.classList.add('cal-today'); }}
    grid.appendChild(el);
  }}
}}

function prevMonth() {{
  if (--currentMonth < 0) {{ currentMonth = 11; currentYear--; }}
  renderCalendar();
}}

function nextMonth() {{
  if (++currentMonth > 11) {{ currentMonth = 0; currentYear++; }}
  renderCalendar();
}}

function selectDate(ds) {{
  var gsInput = document.getElementById('gs-input');
  if (gsInput) gsInput.value = '';
  var srPanel = document.getElementById('search-results-panel');
  srPanel.style.display = 'none';
  srPanel.innerHTML = '';
  document.querySelectorAll('.report-section,.view-section').forEach(function(el) {{
    el.style.display = 'none';
  }});
  document.querySelectorAll('.nav-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.getElementById('no-selection').style.display = 'none';
  var fb = document.getElementById('view-filter-bar');
  if (fb) fb.style.display = 'none';
  currentViewId = null;
  const el = document.getElementById('report-' + ds);
  if (el) {{
    el.style.display = 'block';
    selectedDate = ds;
    renderCalendar();
    document.getElementById('main-content').scrollTop = 0;
  }}
}}

function toggleSidebar() {{
  var sb  = document.getElementById('sidebar');
  var btn = document.getElementById('sidebar-toggle');
  sb.classList.toggle('collapsed');
  var collapsed = sb.classList.contains('collapsed');
  btn.textContent = collapsed ? '▶' : '◀';
  btn.title = collapsed ? 'サイドバーを展開' : 'サイドバーを折りたたむ';
}}

function globalSearch(q) {{
  q = q.trim();
  var panel = document.getElementById('search-results-panel');
  if (!q) {{
    panel.style.display = 'none';
    panel.innerHTML = '';
    document.querySelectorAll('.view-section').forEach(function(el) {{ el.style.display = 'none'; }});
    if (selectedDate) {{
      document.getElementById('report-' + selectedDate).style.display = 'block';
    }} else {{
      document.getElementById('no-selection').style.display = '';
    }}
    return;
  }}
  var ql = q.toLowerCase();
  document.querySelectorAll('.report-section,.view-section').forEach(function(el) {{ el.style.display = 'none'; }});
  document.querySelectorAll('.nav-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.getElementById('no-selection').style.display = 'none';
  var fb = document.getElementById('view-filter-bar');
  if (fb) fb.style.display = 'none';
  currentViewId = null;

  var allResults = [];
  document.querySelectorAll('.report-section').forEach(function(section) {{
    var dateStr = section.id.replace('report-', '');
    var rankEl = section;
    var sectionRows = [];
    var heading = '';
    var walker = document.createTreeWalker(rankEl, NodeFilter.SHOW_ELEMENT, {{
      acceptNode: function(node) {{
        if (node.classList && node.classList.contains('detail-row'))
          return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }}
    }});
    var node;
    while ((node = walker.nextNode())) {{
      if (/^H[2-4]$/.test(node.tagName)) {{
        heading = node.textContent.trim();
      }} else if (node.tagName === 'TR') {{
        var cells = node.cells;
        if (!cells || cells.length === 0) continue;
        if (cells[0].tagName === 'TH') continue;
        if (node.textContent.toLowerCase().includes(ql)) {{
          var tbl = node.closest('table');
          var headerHtml = (tbl && tbl.rows[0]) ? tbl.rows[0].outerHTML : '';
          sectionRows.push({{heading: heading, headerHtml: headerHtml, rowHtml: node.outerHTML}});
        }}
      }}
    }}
    if (sectionRows.length > 0) allResults.push({{dateStr: dateStr, rows: sectionRows}});
  }});

  allResults.sort(function(a, b) {{ return b.dateStr.localeCompare(a.dateStr); }});
  var total = allResults.reduce(function(s, r) {{ return s + r.rows.length; }}, 0);
  var html = '<p class="sr-summary">「' + q + '」&nbsp; ' + total + '件 / ' + allResults.length + '日分</p>';

  allResults.forEach(function(dr) {{
    html += '<div class="sr-date" onclick="selectDate(\\'' + dr.dateStr + '\\')">'
          + '📅 ' + dr.dateStr + '</div>';
    var byH = {{}};
    var headingOrder = [];
    dr.rows.forEach(function(r) {{
      if (!byH[r.heading]) {{
        byH[r.heading] = {{headerHtml: r.headerHtml, rows: []}};
        headingOrder.push(r.heading);
      }}
      byH[r.heading].rows.push(r.rowHtml);
    }});
    headingOrder.forEach(function(h) {{
      var g = byH[h];
      html += '<div class="sr-section">' + (h || '-') + '</div>'
            + '<table class="inner-table" style="margin-bottom:6px">'
            + g.headerHtml + g.rows.join('') + '</table>';
    }});
  }});

  if (!total) html += '<p style="color:#aaa;padding:1em 0">該当なし</p>';
  panel.innerHTML = html;
  panel.style.display = 'block';
}}

renderCalendar();
</script>
</body>
</html>"""

    dash_path = BASE_DIR / "index.html"
    dash_path.write_text(dashboard_html, encoding="utf-8")
    print(f"✅ ダッシュボード更新完了: {dash_path}")


# ---- main ------------------------------------------------------------

def main() -> None:
    target_date = resolve_date(sys.argv[1] if len(sys.argv) >= 2 else None)
    print(f"📊 分析対象日: {target_date}")

    cdr, stf, ext, ring = load_data()
    html = analyze(target_date, cdr, stf, ext, ring)

    html_path = DAILY_DIR / f"{target_date}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ HTMLレポート保存完了: {html_path}")
    build_dashboard()


if __name__ == "__main__":
    main()
