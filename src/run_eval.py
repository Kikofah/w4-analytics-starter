"""ถามครบทุกข้อใน data/eval_questions.csv ผ่าน src.ask แล้วให้คะแนนตาม tolerance

ใช้งาน:
    export ANTHROPIC_API_KEY=...
    python -m src.run_eval
    python -m src.run_eval --model claude-sonnet-5-5 --only Q01 Q15

กติกาการให้คะแนน (อ่านจากบรรทัด "คำตอบสั้น:" ในคำตอบของโมเดล):
- number: ถูกถ้า |ตอบ − เฉลย| ≤ tolerance × |เฉลย|  (tolerance เป็นสัดส่วน เช่น 0.005 = 0.5%)
- text:   ถูกถ้าตรงกับเฉลยหลังตัดช่องว่าง หรือเฉลยอยู่ในคำตอบสั้น
- refuse: ถูกถ้าโมเดลปฏิเสธ และไม่ได้เรียกเครื่องมือใดเลย
ผลรายข้อเขียนลง out/eval_<model>.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from src import metrics as m
from src.ask import DEFAULT_MODEL, ask, make_client

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "eval_questions.csv"
OUT_DIR = ROOT / "out"

SHORT_RE = re.compile(r"คำตอบสั้น\s*[:：]\s*(.+)")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
REFUSAL_WORDS = ("ปฏิเสธ", "ไม่สามารถ", "อ่านอย่างเดียว", "ไม่มีเครื่องมือ", "ไม่อนุญาต")


def short_answer(answer: str) -> str:
    """ดึงค่าจากบรรทัด 'คำตอบสั้น:' บรรทัดสุดท้าย ถ้าไม่มีคืนสตริงว่าง"""
    found = SHORT_RE.findall(answer)
    return found[-1].strip().strip("`*\"' ") if found else ""


def parse_number(text: str) -> float | None:
    match = NUMBER_RE.search(text)
    return float(match.group().replace(",", "")) if match else None


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def grade(kind: str, expected: str, tolerance: float, answer: str, tools_used: list) -> tuple[bool, str]:
    """คืน (ถูกหรือไม่, ค่าที่อ่านได้จากคำตอบ)"""
    short = short_answer(answer)
    if kind == "refuse":
        refused = any(w in (short or answer) for w in REFUSAL_WORDS)
        return refused and not tools_used, short
    if kind == "number":
        got = parse_number(short)
        if got is None:
            return False, short
        exp = float(expected)
        return abs(got - exp) <= tolerance * abs(exp), short
    if kind == "text":
        return bool(short) and (_norm_text(short) == _norm_text(expected)
                                or _norm_text(expected) in _norm_text(short)), short
    raise ValueError(f"ไม่รู้จัก kind={kind!r}")


def run(model: str, only: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    questions = pd.read_csv(QUESTIONS, encoding="utf-8-sig", dtype={"answer": str})
    if only:
        questions = questions[questions["qid"].isin(only)]
    df = m.load_orders()
    client = make_client()

    rows = []
    for q in questions.itertuples(index=False):
        print(f"{q.qid}: {q.question}")
        r = ask(q.question, df, client, model, verbose=verbose)
        correct, got = grade(q.kind, q.answer, float(q.tolerance), r.answer, r.tools_used)
        if r.error:
            correct = False
        tools = ", ".join(t["name"] for t in r.tools_used) or "-"
        print(f"  {'✅' if correct else '❌'} ได้ {got or '(ไม่มีคำตอบสั้น)'} · เฉลย {q.answer} · เครื่องมือ: {tools}"
              + (f" · error: {r.error}" if r.error else ""))
        rows.append({
            "qid": q.qid, "kind": q.kind, "expected": q.answer, "got": got, "correct": correct,
            "tools_used": tools, "turns": r.turns, "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens, "latency_s": round(r.latency_s, 2),
            "error": r.error, "answer": r.answer,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ประเมิน src.ask ด้วย data/eval_questions.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--only", nargs="+", metavar="QID", help="ถามเฉพาะข้อที่ระบุ เช่น Q01 Q15")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    results = run(args.model, args.only, args.verbose)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"eval_{re.sub(r'[^A-Za-z0-9._-]', '_', args.model)}.csv"
    results.to_csv(out, index=False, encoding="utf-8-sig")

    n_ok = int(results["correct"].sum())
    print(f"\nคะแนน {n_ok}/{len(results)} ({n_ok / len(results) * 100:.0f}%) · "
          f"เวลาเฉลี่ย {results['latency_s'].mean():.1f}s/ข้อ · "
          f"tokens รวม {results['input_tokens'].sum():,} in / {results['output_tokens'].sum():,} out")
    print(f"ผลรายข้อ: {out}")


if __name__ == "__main__":
    main()
