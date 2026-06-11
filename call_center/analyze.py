"""
Call center daily analysis script.

Usage:
    python -X utf8 call_center/analyze.py [YYYY-MM-DD]

Output:
    call_center/reports/YYYY-MM-DD.md
    call_center/reports/YYYY-MM-DD.html

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
        hourly.append({"時間帯": f"{h}時台", "着信": calls, "応答": ans, "未応答": mis,
                       "応答率": r, "判定": judge, "active_ops": active_cnt,
                       "avg_talk_secs": avg_talk_h,
                       "prev_d_rate": pd_s["rate"],   "prev_w_rate":   pw_s["rate"],
                       "prev_d_total": pd_s["total"], "prev_w_total":  pw_s["total"],
                       "prev_d_got":   pd_s["got"],   "prev_w_got":    pw_s["got"],
                       "prev_d_missed":pd_s["missed"],"prev_w_missed": pw_s["missed"],
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
        ext_daily_list.append({
            "ext":    ext_key,
            "name":   str(row["案件名"]),
            "type":   str(row["内線種別"]),
            "total":  int(row["着信"]),
            "got":    int(row["応答"]),
            "missed": int(row["未応答"]),
            "rate":   float(row["応答率"]),
            "kpi":    kpi_label(float(row["応答率"])),
            "h":      h_total,
            "g":      h_got,
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

    # --- 前日・前週比（集計値） ---
    prev_day_s  = _day_stats(cdr, stf, prev_day_date,  in_ext_types)
    prev_week_s = _day_stats(cdr, stf, prev_week_date, in_ext_types)

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
        hourly=hourly,
        miss_counts=miss_counts,
        ext_best5=ext_best5, ext_worst5=ext_worst5, ext_by_volume=ext_by_volume,
        ext_special=ext_special, ext_daily=ext_daily_list,
        op_cph_best5=op_cph_best5, op_cph_worst5=op_cph_worst5,
        op_rate_best5=op_rate_best5, op_rate_worst5=op_rate_worst5,
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
    ps    = d["prev_day_s"]
    pw    = d["prev_week_s"]
    total = d["total"]

    # --- 需要要因 ---
    if ps and (ps.get("total") or 0) > 0:
        ratio = (total - ps["total"]) / ps["total"]
        if abs(ratio) >= 0.10:
            factors["需要"].append(
                f"着信数が前日比 {round(ratio * 100, 1):+.1f}%（{'増加' if ratio > 0 else '減少'}）")
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
        if ps and (ps.get("avg_cph") or 0) > 0:
            ratio = (cph - ps["avg_cph"]) / ps["avg_cph"]
            if abs(ratio) >= 0.20:
                factors["生産性"].append(
                    f"CPHが前日比 {round(ratio * 100, 1):+.1f}%（{'上昇' if ratio > 0 else '低下'}）")
    secs = d.get("avg_talk_secs")
    if secs is not None:
        if secs > 300:
            factors["生産性"].append(f"平均通話時間が長い（{fmt_seconds(secs)}）")
        if ps and (ps.get("avg_talk") or 0) > 0:
            ratio = (secs - ps["avg_talk"]) / ps["avg_talk"]
            if abs(ratio) >= 0.20:
                factors["生産性"].append(
                    f"平均通話時間が前日比 {round(ratio * 100, 1):+.1f}%（{'増加' if ratio > 0 else '減少'}）")

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


def build_html(d: dict) -> str:
    ps  = d["prev_day_s"]
    pw  = d["prev_week_s"]

    def dcell(cur, prev_d_key: str, unit: str = "件", is_rate: bool = False,
              use_float: bool = False) -> tuple[str, str]:
        pd_val = ps[prev_d_key] if (ps and prev_d_key in ps) else None
        pw_val = pw[prev_d_key] if (pw and prev_d_key in pw) else None
        return (_diff_str(cur, pd_val, unit, is_rate, use_float),
                _diff_str(cur, pw_val, unit, is_rate, use_float))

    def srow(label: str, cur_disp: str, cur_raw, key: str, unit: str = "件",
             is_rate: bool = False, use_float: bool = False) -> str:
        day_s, week_s = dcell(cur_raw, key, unit, is_rate, use_float)
        return (f'<tr><td style="text-align:left">{label}</td><td><strong>{cur_disp}</strong></td>'
                f"<td>{day_s}</td><td>{week_s}</td></tr>")

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

    def _h_tot_diff(cur, key: str, unit: str = "件", is_rate: bool = False) -> tuple[str, str]:
        pd_v = ps[key] if (ps and key in ps) else None
        pw_v = pw[key] if (pw and key in pw) else None
        return (f"<td>{_diff_str(cur, pd_v, unit, is_rate)}</td>",
                f"<td>{_diff_str(cur, pw_v, unit, is_rate)}</td>")

    tot_d_total,  tot_w_total  = _h_tot_diff(d["total"],  "total")
    tot_d_got,    tot_w_got    = _h_tot_diff(d["got"],    "got")
    tot_d_missed, tot_w_missed = _h_tot_diff(d["missed"], "missed")
    tot_d_rate,   tot_w_rate   = _h_tot_diff(d["rate"],   "rate", "%", True)

    if any(r["着信"] > 0 for r in d["hourly"]):
        hourly_table_html = (
            '<div class="hourly-wrap"><table>'
            f'<tr><th></th>{h_headers}</tr>'
            + _hrow("着信数",    f"<td>{d['total']}</td>"   + _tds([d["hourly"][h]["着信"]   for h in all_hours]))
            + _hrow("　前日比",  tot_d_total  + _tds([_diff_str(d["hourly"][h]["着信"],   d["hourly"][h]["prev_d_total"],  "件") for h in all_hours]), "comp-row")
            + _hrow("　前週比",  tot_w_total  + _tds([_diff_str(d["hourly"][h]["着信"],   d["hourly"][h]["prev_w_total"],  "件") for h in all_hours]), "comp-row")
            + _hrow("応答数",    f"<td>{d['got']}</td>"     + _tds([d["hourly"][h]["応答"]   for h in all_hours]))
            + _hrow("　前日比",  tot_d_got    + _tds([_diff_str(d["hourly"][h]["応答"],   d["hourly"][h]["prev_d_got"],    "件") for h in all_hours]), "comp-row")
            + _hrow("　前週比",  tot_w_got    + _tds([_diff_str(d["hourly"][h]["応答"],   d["hourly"][h]["prev_w_got"],    "件") for h in all_hours]), "comp-row")
            + _hrow("未応答数",  f"<td>{d['missed']}</td>"  + _tds([d["hourly"][h]["未応答"]  for h in all_hours]))
            + _hrow("　前日比",  tot_d_missed + _tds([_diff_str(d["hourly"][h]["未応答"], d["hourly"][h]["prev_d_missed"], "件") for h in all_hours]), "comp-row")
            + _hrow("　前週比",  tot_w_missed + _tds([_diff_str(d["hourly"][h]["未応答"], d["hourly"][h]["prev_w_missed"], "件") for h in all_hours]), "comp-row")
            + _hrow("応答率",    _rate_td(d["rate"]) + "".join(_rate_td(d["hourly"][h]["応答率"]) for h in all_hours))
            + _hrow("　前日比",  tot_d_rate   + _tds([_diff_str(d["hourly"][h]["応答率"], d["hourly"][h]["prev_d_rate"],   "%", is_rate=True) for h in all_hours]), "comp-row")
            + _hrow("　前週比",  tot_w_rate   + _tds([_diff_str(d["hourly"][h]["応答率"], d["hourly"][h]["prev_w_rate"],   "%", is_rate=True) for h in all_hours]), "comp-row")
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
        _pr = round((_pt - _pm) / _pt * 100, 1) if _pt > 0 else 100.0
        _ab_stats.append({"name": _pn, "total": _pt, "missed": _pm, "rate": _pr})

    _ab_stats = [_p for _p in _ab_stats if _p["rate"] < 90]
    _ab_stats.sort(key=lambda x: (x["rate"], -x["total"]))

    if _ab_stats:
        ab_ranking_html = '<h3>特定案件ランキング（[A]/[B]・応答率90%未満）</h3>\n'
        ab_ranking_html += (
            '<table><tr>'
            '<th style="text-align:left">案件名</th>'
            '<th>着信</th><th>未応答</th><th>応答率</th>'
            '</tr>\n'
        )
        for _p in _ab_stats:
            _rc = rate_class(_p["rate"])
            ab_ranking_html += (
                f'<tr><td style="text-align:left">{_p["name"]}</td>'
                f'<td>{_p["total"]}</td><td>{_p["missed"]}</td>'
                f'<td class="{_rc}">{_p["rate"]}%</td></tr>\n'
            )
        ab_ranking_html += '</table>\n'
        for _p in _ab_stats:
            _pn = _p["name"]
            ab_ranking_html += f'<h4 style="margin-top:1em">{_pn}</h4>\n'
            _hg: dict = {}
            for _h, _c in _ab_by_proj[_pn]:
                if _h not in _hg:
                    _hg[_h] = []
                _hg[_h].append(_c)
            ab_ranking_html += (
                '<table><tr><th>時間帯</th><th>着信</th><th>未応答</th>'
                '<th>稼働OP数</th><th style="text-align:left">取れなかった理由</th></tr>\n'
            )
            for _h in sorted(_hg):
                _clist = _hg[_h]
                _th = len(_clist)
                _mh = sum(1 for _c in _clist if _c["結果"] == "未応答")
                _oh = d["hourly"][_h]["active_ops"]
                _rs = _miss_by_ph.get((_pn, _h), [])
                if _mh == 0:
                    _rstr = "-"
                elif _rs:
                    _rcnt = Counter(_rs)
                    _rstr = "、".join(
                        f"{_r}({_cnt}件)" for _r, _cnt in sorted(_rcnt.items())
                    )
                else:
                    _rstr = "確認できません"
                ab_ranking_html += (
                    f'<tr><td>{_h}時台</td><td>{_th}</td><td>{_mh}</td>'
                    f'<td>{_oh}</td>'
                    f'<td style="text-align:left">{_rstr}</td></tr>\n'
                )
            ab_ranking_html += '</table>\n'
    else:
        ab_ranking_html = ""

    total_day_d,  total_week_d  = dcell(d["total"],          "total",    "件")
    got_day_d,    got_week_d    = dcell(d["got"],             "got",      "件")
    missed_day_d, missed_week_d = dcell(d["missed"],          "missed",   "件")
    rate_day_d,   rate_week_d   = dcell(d["rate"],            "rate",     "%",  True)
    work_day_d,   work_week_d   = dcell(d["total_work_hrs"],  "work_hrs", "h",  False, True)
    cph_day_d,    cph_week_d    = dcell(cph_raw,              "avg_cph",  "",   False, True)
    talk_day_d,   talk_week_d   = dcell(talk_raw,             "avg_talk", "秒")

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
            "total_day":  total_day_d,  "total_week":  total_week_d,
            "got_day":    got_day_d,    "got_week":    got_week_d,
            "missed_day": missed_day_d, "missed_week": missed_week_d,
            "rate_day":   rate_day_d,   "rate_week":   rate_week_d,
            "work_day":   work_day_d,   "work_week":   work_week_d,
            "cph_day":    cph_day_d,    "cph_week":    cph_week_d,
            "talk_day":   talk_day_d,   "talk_week":   talk_week_d,
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
<div>{_html_table(["KPI", "実績", "前日比", "前週同曜日比"], summary_rows)}</div>
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
function renderMonthView(el, isThis) {
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
  if (!rows.length) {
    el.innerHTML = '<h2>' + title + '</h2><p style="color:#aaa">データなし</p>';
    return;
  }
  var days = rows.length;
  var sumT=0, sumG=0, sumM=0, sumW=0, cphSum=0, cphN=0, talkSum=0, talkN=0, rateSum=0, rateN=0;
  rows.forEach(function(r) {
    sumT += r.total; sumG += r.got; sumM += r.missed; sumW += r.work_hrs || 0;
    if (r.avg_cph != null)      { cphSum  += r.avg_cph;      cphN++;  }
    if (r.avg_talk_secs != null){ talkSum += r.avg_talk_secs; talkN++; }
    if (r.rate != null)          { rateSum += r.rate;          rateN++; }
  });
  var rateTotal = sumT > 0 ? Math.round(sumG / sumT * 1000) / 10 : null;
  var rateAvg   = rateN  > 0 ? Math.round(rateSum  / rateN  * 10) / 10 : null;
  var avgCph    = cphN   > 0 ? Math.round(cphSum   / cphN   * 10) / 10 : null;
  var avgTalk   = talkN  > 0 ? Math.round(talkSum  / talkN)            : null;
  var rc  = rateCls(rateTotal), rc2 = rateCls(rateAvg), cc = cphCls(avgCph);
  var period = rows[0].date + ' 〜 ' + rows[rows.length - 1].date;
  var fl = extFilterLabel();
  var titleSuffix = fl ? ' <span style="font-size:0.75em;background:#e8f0fe;color:#1a73e8;border-radius:3px;padding:2px 6px">' + fl + '</span>' : '';
  var html = '<h2>' + title + titleSuffix + '</h2><p style="color:#777;font-size:0.9em">対象期間: ' + period + '（' + days + '日）</p>';
  if (fl) html += '<p style="color:#888;font-size:0.85em">※ 案件絞り込み中。稼働時間・CPH・通話時間は全体値です。</p>';
  html += '<h3>KPI</h3><table>' +
    '<tr><th style="text-align:left">指標</th><th>合計</th><th>平均（1日あたり）</th></tr>' +
    '<tr><td style="text-align:left">稼働日数</td><td>' + days + '日</td><td>-</td></tr>' +
    '<tr><td style="text-align:left">着信数</td><td>' + sumT + '件</td><td>' + Math.round(sumT/days*10)/10 + '件</td></tr>' +
    '<tr><td style="text-align:left">応答数</td><td>' + sumG + '件</td><td>' + Math.round(sumG/days*10)/10 + '件</td></tr>' +
    '<tr><td style="text-align:left">未応答数</td><td>' + sumM + '件</td><td>' + Math.round(sumM/days*10)/10 + '件</td></tr>' +
    '<tr><td style="text-align:left">応答率</td>' +
      '<td class="' + rc  + '">' + (rateTotal != null ? rateTotal + '%' : '-') + '</td>' +
      '<td class="' + rc2 + '">' + (rateAvg   != null ? rateAvg   + '%' : '-') + '</td></tr>' +
    (fl ? '' :
    '<tr><td style="text-align:left">稼働時間</td><td>' + sumW.toFixed(1) + 'h</td><td>' + (sumW/days).toFixed(1) + 'h</td></tr>' +
    '<tr><td style="text-align:left">平均CPH</td><td>-</td><td class="' + cc + '">' + (avgCph != null ? avgCph : '-') + '</td></tr>' +
    '<tr><td style="text-align:left">平均通話時間</td><td>-</td><td>' + fmtSecs(avgTalk) + '</td></tr>') +
    '</table>';
  var wDays = ['月','火','水','木','金','土','日'];
  var wBuckets = {};
  wDays.forEach(function(d) { wBuckets[d] = []; });
  rows.forEach(function(r) { if (r.weekday && wBuckets[r.weekday] !== undefined) wBuckets[r.weekday].push(r); });
  html += '<h3>曜日別集計</h3><table>' +
    '<tr><th>曜日</th><th>日数</th><th>着信合計</th><th>平均着信</th><th>未応答合計</th><th>平均応答率</th><th>平均CPH</th></tr>';
  wDays.forEach(function(day) {
    var rs = wBuckets[day];
    if (!rs.length) {
      html += '<tr><td>' + day + '</td><td>0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>';
      return;
    }
    var n = rs.length;
    var wT = rs.reduce(function(s,r){return s+r.total;},0);
    var wG = rs.reduce(function(s,r){return s+r.got;},0);
    var wM = rs.reduce(function(s,r){return s+r.missed;},0);
    var wRate = wT > 0 ? Math.round(wG/wT*1000)/10 : null;
    var wCArr = rs.filter(function(r){return r.avg_cph != null;});
    var wCph = wCArr.length ? Math.round(wCArr.reduce(function(s,r){return s+r.avg_cph;},0)/wCArr.length*10)/10 : null;
    html += '<tr><td>' + day + '</td><td>' + n + '</td><td>' + wT + '</td><td>' + Math.round(wT/n*10)/10 + '</td><td>' + wM + '</td>' +
      '<td class="' + rateCls(wRate) + '">' + (wRate != null ? wRate + '%' : '-') + '</td>' +
      '<td class="' + cphCls(wCph)  + '">' + (wCph  != null ? wCph  : '-') + '</td></tr>';
  });
  html += '</table>';
  html += '<h3>時間帯別累計</h3><table>' +
    '<tr><th>時間帯</th><th>着信合計</th><th>平均着信</th><th>応答合計</th><th>未応答合計</th><th>応答率</th><th>平均人員</th><th>平均通話時間</th></tr>';
  for (var h = 0; h < 24; h++) {
    var hT=0, hG=0, hM=0, hOps=0, hTalkS=0, hTalkN=0, hDays=0;
    rows.forEach(function(r) {
      if (!r.hourly || !r.hourly[h]) return;
      var hh = r.hourly[h];
      hT += hh.total||0; hG += hh.got||0; hM += hh.missed||0; hOps += hh.ops||0;
      if (hh.avg_talk != null) { hTalkS += hh.avg_talk; hTalkN++; }
      if ((hh.total||0) > 0) hDays++;
    });
    var hRate = hT > 0 ? Math.round(hG/hT*1000)/10 : null;
    html += '<tr><td>' + h + '時台</td><td>' + hT + '</td>' +
      '<td>' + (hDays > 0 ? Math.round(hT/hDays*10)/10 : '-') + '</td>' +
      '<td>' + hG + '</td><td>' + hM + '</td>' +
      '<td class="' + rateCls(hRate) + '">' + (hRate != null ? hRate + '%' : '-') + '</td>' +
      '<td>' + Math.round(hOps/rows.length*10)/10 + '</td>' +
      '<td>' + fmtSecs(hTalkN > 0 ? Math.round(hTalkS/hTalkN) : null) + '</td></tr>';
  }
  html += '</table>';
  html += '<h3>日別詳細</h3><table>' +
    '<tr><th>日付</th><th>曜日</th><th>着信</th><th>応答</th><th>未応答</th><th>応答率</th><th>稼働時間</th><th>CPH</th></tr>';
  rows.forEach(function(r) {
    var rc3 = rateCls(r.rate), cc3 = cphCls(r.avg_cph);
    html += '<tr><td>' + r.date + '</td><td>' + (r.weekday || '') + '</td>' +
      '<td>' + r.total + '</td><td>' + r.got + '</td><td>' + r.missed + '</td>' +
      '<td class="' + rc3 + '">' + (r.rate != null ? r.rate + '%' : '-') + '</td>' +
      '<td>' + (r.work_hrs != null ? r.work_hrs.toFixed(1) + 'h' : '-') + '</td>' +
      '<td class="' + cc3 + '">' + (r.avg_cph != null ? r.avg_cph : '-') + '</td></tr>';
  });
  html += '</table>';
  el.innerHTML = html;
}
function renderWeekdayView(el) {
  var rows = getFilteredDays();
  if (!rows.length) {
    el.innerHTML = '<h2>曜日別累計</h2><p style="color:#aaa">データなし</p>';
    return;
  }
  var allDates = rows.map(function(r) { return r.date; }).sort();
  var period = allDates[0] + ' 〜 ' + allDates[allDates.length - 1];
  var days = ['月','火','水','木','金','土','日'];
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
    cutoff = today - timedelta(days=60)

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

    dash_path = REPORTS_DIR / "dashboard.html"
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
