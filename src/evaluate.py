"""ประเมินผลการจัดหมวดหมู่ตั๋วของแต่ละโมเดลเทียบกับ gold

ใช้งาน:
    python -m src.evaluate                      # ทุกไฟล์ out/pred_*.csv
    python -m src.evaluate out/pred_claude-opus-5-5.csv
    python -m src.evaluate --dev                # ทุกไฟล์ out/dev_pred_*.csv เทียบกับ dev gold

ผลลัพธ์: พิมพ์สรุปออกจอ และเขียน reports/model_comparison.md
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "tickets_test_gold.csv"
DEV_GOLD = ROOT / "data" / "tickets_dev_gold.csv"
OUT_DIR = ROOT / "out"
REPORT = ROOT / "reports" / "model_comparison.md"
DEV_REPORT = ROOT / "reports" / "model_comparison_dev.md"

FIELDS = ["category", "sentiment", "urgency", "order_id_mentioned", "contains_pii"]
CATEGORIES = ["ส่งช้า", "สินค้าชำรุด", "ได้รับสินค้าผิด", "ขอคืนเงิน",
              "เปลี่ยนแปลงคำสั่งซื้อ", "ชำระเงินและโปรโมชั่น", "สอบถามทั่วไป", "ชมเชย"]
HIGH_URGENCY = "สูง"

# ราคา USD ต่อ 1 ล้าน token (input, output) ตามราคา API ของ Anthropic ณ ต.ค. 2026
# ตรวจราคาล่าสุดที่ https://platform.claude.com/docs/en/about-claude/pricing ก่อนใช้ตัวเลขจริง
PRICES_PER_MTOK = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_WRITE_MULT = 1.25  # cache write (TTL 5 นาที) = 1.25 เท่าของราคา input
CACHE_READ_MULT = 0.10   # cache read = 0.1 เท่าของราคา input


@dataclass
class ModelResult:
    model: str
    n_gold: int
    n_pred: int            # จำนวนตั๋ว gold ที่มีในไฟล์ผล
    n_error: int           # ตั๋วที่เรียก API ไม่สำเร็จ (นับเป็นตอบผิด)
    accuracy: dict[str, float]
    exact_match: float     # ถูกครบทั้ง 5 ฟิลด์
    high_urgency_recall: float
    n_high_urgency: int
    avg_latency_s: float
    cost_per_1000_usd: float | None
    confusion: dict[str, pd.DataFrame] = field(default_factory=dict)


# ---------- การทำให้ค่าเทียบกันได้ ----------

def _norm_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _norm_order_id(s: pd.Series) -> pd.Series:
    return _norm_str(s).str.upper()


def _norm_bool(s: pd.Series) -> pd.Series:
    mapping = {"true": "True", "false": "False", "1": "True", "0": "False"}
    return _norm_str(s).str.lower().map(mapping).fillna("")


NORMALIZERS = {
    "category": _norm_str,
    "sentiment": _norm_str,
    "urgency": _norm_str,
    "order_id_mentioned": _norm_order_id,
    "contains_pii": _norm_bool,
}


def load_gold(path: Path | str = GOLD) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str)


def load_pred(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str)


# ---------- ตัวชี้วัด ----------

def cost_usd(model: str, pred: pd.DataFrame) -> pd.Series | None:
    """ต้นทุนต่อใบ (USD) จาก usage คืน None ถ้าไม่รู้ราคาโมเดล"""
    if model not in PRICES_PER_MTOK:
        return None
    price_in, price_out = PRICES_PER_MTOK[model]

    def col(name: str) -> pd.Series:
        if name not in pred:
            return pd.Series(0.0, index=pred.index)
        return pd.to_numeric(pred[name], errors="coerce").fillna(0)

    return (col("input_tokens") * price_in
            + col("cache_creation_input_tokens") * price_in * CACHE_WRITE_MULT
            + col("cache_read_input_tokens") * price_in * CACHE_READ_MULT
            + col("output_tokens") * price_out) / 1_000_000


def confusion_matrix(gold: pd.Series, pred: pd.Series, labels: list[str]) -> pd.DataFrame:
    """แถว = gold, คอลัมน์ = pred (มีคอลัมน์ '(ไม่มีคำตอบ)' สำหรับใบที่ error หรือค่านอก label)"""
    other = "(ไม่มีคำตอบ)"
    pred = pred.where(pred.isin(labels), other)
    cm = pd.crosstab(pd.Categorical(gold, categories=labels),
                     pd.Categorical(pred, categories=labels + [other]),
                     rownames=["gold"], colnames=["pred"], dropna=False)
    if cm[other].sum() == 0:
        cm = cm.drop(columns=other)
    return cm


def evaluate(pred: pd.DataFrame, gold: pd.DataFrame, model: str | None = None) -> ModelResult:
    if model is None:
        model = str(pred["model"].dropna().iloc[0]) if "model" in pred and pred["model"].notna().any() else "unknown"

    pred = pred.drop_duplicates("ticket_id", keep="last")
    # ใช้ gold เป็นฐาน: ใบที่ไม่มีในไฟล์ผลไม่นับ (ดู coverage), ใบที่ error นับเป็นตอบผิด
    merged = gold.merge(pred, on="ticket_id", how="inner", suffixes=("_gold", "_pred"))
    if merged.empty:
        raise ValueError(f"{model}: ไม่มีตั๋วใน gold ที่ตรงกับไฟล์ผลเลย "
                         "ให้รัน python -m src.classify <model> --gold (หรือ --dev สำหรับชุด dev) ก่อน")

    has_error = _norm_str(merged["error"]) != "" if "error" in merged else pd.Series(False, index=merged.index)

    correct = {}
    norm = {}
    for f in FIELDS:
        g = NORMALIZERS[f](merged[f"{f}_gold"])
        p = NORMALIZERS[f](merged[f"{f}_pred"]).where(~has_error, "")
        norm[f] = (g, p)
        correct[f] = (g == p) & ~has_error

    accuracy = {f: float(correct[f].mean()) for f in FIELDS}
    exact = pd.concat(correct.values(), axis=1).all(axis=1)

    g_urg, p_urg = norm["urgency"]
    high = g_urg == HIGH_URGENCY
    recall = float((p_urg[high] == HIGH_URGENCY).mean()) if high.any() else float("nan")

    ok = pred[pred["ticket_id"].isin(merged["ticket_id"]) & ~pred["ticket_id"].isin(merged.loc[has_error, "ticket_id"])]
    latency = pd.to_numeric(ok.get("latency_s"), errors="coerce").mean() if "latency_s" in ok else float("nan")
    costs = cost_usd(model, ok)
    cost_1000 = float(costs.mean() * 1000) if costs is not None and len(ok) else None

    confusion = {
        "category": confusion_matrix(*norm["category"], CATEGORIES),
        "urgency": confusion_matrix(*norm["urgency"], sorted(set(norm["urgency"][0]) - {""})),
        "sentiment": confusion_matrix(*norm["sentiment"], ["บวก", "กลาง", "ลบ"]),
    }

    return ModelResult(
        model=model, n_gold=len(gold), n_pred=len(merged), n_error=int(has_error.sum()),
        accuracy=accuracy, exact_match=float(exact.mean()),
        high_urgency_recall=recall, n_high_urgency=int(high.sum()),
        avg_latency_s=float(latency), cost_per_1000_usd=cost_1000, confusion=confusion,
    )


# ---------- รายงาน ----------

def _pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x * 100:.1f}%"


def _md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or "", *map(str, df.columns)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx), *map(str, row.tolist())]) + " |")
    return "\n".join(lines)


def build_comparison(results: list[ModelResult], gold_name: str = "data/tickets_test_gold.csv") -> str:
    lines = [
        "# เปรียบเทียบโมเดลจัดหมวดหมู่ตั๋ว",
        "",
        f"ชุดทดสอบ: `{gold_name}` ({results[0].n_gold} ใบ) · "
        "ตั๋วที่เรียก API ไม่สำเร็จนับเป็นตอบผิด",
        "",
        "## สรุป",
        "",
        "| โมเดล | ใบที่ประเมิน | error | " + " | ".join(f"acc {f}" for f in FIELDS)
        + " | ถูกครบ 5 ฟิลด์ | recall urgency=สูง | เวลาเฉลี่ย (วินาที) | ต้นทุน / 1,000 ใบ (USD) |",
        "|---|---:|---:|" + "---:|" * len(FIELDS) + "---:|---:|---:|---:|",
    ]
    for r in results:
        cost = "-" if r.cost_per_1000_usd is None else f"${r.cost_per_1000_usd:,.2f}"
        lines.append(
            f"| `{r.model}` | {r.n_pred}/{r.n_gold} | {r.n_error} | "
            + " | ".join(_pct(r.accuracy[f]) for f in FIELDS)
            + f" | {_pct(r.exact_match)} | {_pct(r.high_urgency_recall)} (n={r.n_high_urgency})"
            + f" | {r.avg_latency_s:.2f} | {cost} |"
        )

    partial = [r.model for r in results if r.n_pred < r.n_gold]
    if partial:
        lines += ["", f"> ⚠️ ประเมินไม่ครบ {results[0].n_gold} ใบ: {', '.join(f'`{m}`' for m in partial)} "
                      "ตัวเลขยังเทียบกันตรง ๆ ไม่ได้ ให้รัน `python -m src.classify <model> --gold` (ชุด dev ใช้ `--dev`)"]
    unpriced = [r.model for r in results if r.cost_per_1000_usd is None]
    if unpriced:
        lines += ["", f"> ไม่มีราคาของ {', '.join(f'`{m}`' for m in unpriced)} ใน `PRICES_PER_MTOK`"]

    lines += ["", "ต้นทุนคิดจาก usage จริงต่อใบ × ราคา API ใน `src/evaluate.py` "
                  "(cache write 1.25×, cache read 0.1× ของราคา input)", ""]

    for r in results:
        lines += [f"## Confusion matrix · `{r.model}`", "", "แถว = gold · คอลัมน์ = คำตอบของโมเดล", ""]
        for name, cm in r.confusion.items():
            lines += [f"### {name}", "", _md_table(cm), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ประเมินผลการจัดหมวดหมู่ตั๋วเทียบกับ gold")
    parser.add_argument("preds", nargs="*", type=Path,
                        help="ไฟล์ผล (ค่าเริ่มต้น: ทุกไฟล์ out/pred_*.csv)")
    parser.add_argument("--dev", action="store_true",
                        help="ประเมินชุด dev: อ่าน out/dev_pred_*.csv เทียบ tickets_dev_gold.csv "
                             "เขียน reports/model_comparison_dev.md")
    parser.add_argument("--gold", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)

    gold_path = args.gold or (DEV_GOLD if args.dev else GOLD)
    output = args.output or (DEV_REPORT if args.dev else REPORT)
    paths = args.preds or sorted(OUT_DIR.glob("dev_pred_*.csv" if args.dev else "pred_*.csv"))
    if not paths:
        flag = "--dev" if args.dev else "--gold"
        parser.exit(1, f"ไม่พบไฟล์ผลใน out/ ให้รัน python -m src.classify <model> {flag} ก่อน\n")

    gold = load_gold(gold_path)
    results = []
    for p in paths:
        try:
            r = evaluate(load_pred(p), gold)
        except ValueError as e:
            print(f"ข้าม {p.name}: {e}")
            continue
        results.append(r)
        acc = " ".join(f"{f}={_pct(v)}" for f, v in r.accuracy.items())
        cost = "-" if r.cost_per_1000_usd is None else f"${r.cost_per_1000_usd:.2f}"
        print(f"{r.model}: {r.n_pred}/{r.n_gold} ใบ · {acc} · recall สูง={_pct(r.high_urgency_recall)} "
              f"· {r.avg_latency_s:.2f}s/ใบ · {cost}/1,000 ใบ")

    if not results:
        parser.exit(1, "ไม่มีไฟล์ผลที่ประเมินได้\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    gold_name = gold_path.relative_to(ROOT) if gold_path.is_relative_to(ROOT) else gold_path
    output.write_text(build_comparison(results, str(gold_name)), encoding="utf-8")
    print(f"เขียนตารางเทียบแล้ว: {output}")


if __name__ == "__main__":
    main()
