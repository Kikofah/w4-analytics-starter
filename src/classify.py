"""จัดหมวดหมู่ตั๋วร้องเรียนลูกค้าด้วย Claude API

ใช้งาน:
    export ANTHROPIC_API_KEY=...            # อ่าน key จาก environment variable เท่านั้น
    python -m src.classify claude-opus-5-5             # 5 ใบแรก (ค่าเริ่มต้น)
    python -m src.classify claude-opus-5-5 --limit 20
    python -m src.classify claude-opus-5-5 --all
    python -m src.classify claude-opus-5-5 --gold     # เฉพาะ 100 ใบใน gold (ใช้กับ src.evaluate)

ผลลัพธ์: out/pred_<model>.csv หนึ่งแถวต่อตั๋ว พร้อม usage และเวลาที่ใช้ต่อใบ
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd

from src.mask_pii import EMAIL_TOKEN, PHONE_TOKEN, mask_dataframe

ROOT = Path(__file__).resolve().parent.parent
TICKETS = ROOT / "data" / "tickets.csv"
GOLD = ROOT / "data" / "tickets_test_gold.csv"
PROMPT = ROOT / "prompts" / "classify_v3.md"
OUT_DIR = ROOT / "out"

# ข้อความตั้งต้นใน prompts/classify_v3.md ที่ยังไม่ได้วาง prompt จริง
PLACEHOLDER_MARKER = "วาง prompt v3"

CATEGORIES = [
    "ส่งช้า",
    "สินค้าชำรุด",
    "ได้รับสินค้าผิด",
    "ขอคืนเงิน",
    "เปลี่ยนแปลงคำสั่งซื้อ",
    "ชำระเงินและโปรโมชั่น",
    "สอบถามทั่วไป",
    "ชมเชย",
]
SENTIMENTS = ["บวก", "กลาง", "ลบ"]
URGENCIES = ["ปกติ", "สูง"]

LABEL_FIELDS = ["category", "sentiment", "urgency", "order_id_mentioned", "contains_pii"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "sentiment": {"type": "string", "enum": SENTIMENTS},
        "urgency": {"type": "string", "enum": URGENCIES},
        "order_id_mentioned": {
            "type": "string",
            "description": "เลขคำสั่งซื้อที่ลูกค้าระบุ เช่น BS2510210020 ถ้าไม่มีให้เป็นสตริงว่าง",
        },
        "contains_pii": {"type": "boolean"},
    },
    "required": LABEL_FIELDS,
    "additionalProperties": False,
}

OUT_COLUMNS = ["ticket_id", *LABEL_FIELDS, "model", "stop_reason",
               "input_tokens", "output_tokens", "cache_creation_input_tokens",
               "cache_read_input_tokens", "latency_s", "request_id", "error"]


def load_system_prompt(path: Path = PROMPT) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text or PLACEHOLDER_MARKER in text:
        sys.exit(f"{path} ยังเป็นข้อความตั้งต้นอยู่ ให้วาง prompt v3 ลงในไฟล์ก่อนรัน")
    return text


def make_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ไม่พบ ANTHROPIC_API_KEY ให้ตั้งค่าด้วย: export ANTHROPIC_API_KEY=...")
    return anthropic.Anthropic(api_key=api_key)


def classify_one(client: anthropic.Anthropic, model: str, system: str, message: str,
                 effort: str | None = None) -> tuple[dict, anthropic.types.Message, float]:
    """ส่งตั๋วหนึ่งใบ คืน (labels, response, latency_s)"""
    output_config: dict = {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
    if effort:
        output_config["effort"] = effort
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=system,
        messages=[{
            "role": "user",
            # ห่อข้อความลูกค้าไว้ในแท็ก ให้โมเดลถือเป็นข้อมูล ไม่ใช่คำสั่ง
            "content": MASK_NOTE + f"<ticket>\n{message}\n</ticket>",
        }],
        output_config=output_config,
    )
    latency = time.perf_counter() - start

    if response.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text), response, latency


MASK_NOTE = (f"หมายเหตุ: {PHONE_TOKEN} และ {EMAIL_TOKEN} ในข้อความคือเบอร์โทรและอีเมลของลูกค้า "
             "ที่ถูกปิดบังก่อนส่ง (นับเป็นข้อมูลส่วนบุคคล)\n")


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def run(model: str, limit: int | None, effort: str | None = None, gold_only: bool = False) -> Path:
    system = load_system_prompt()
    client = make_client()
    tickets = pd.read_csv(TICKETS, encoding="utf-8-sig", dtype=str)
    if gold_only:
        gold_ids = pd.read_csv(GOLD, encoding="utf-8-sig", dtype=str)["ticket_id"]
        tickets = tickets[tickets["ticket_id"].isin(gold_ids)]
    if limit is not None:
        tickets = tickets.head(limit)

    # ปิดบังเบอร์โทรและอีเมลก่อนส่งเข้า API ทุกครั้ง (policies/04_privacy.md)
    tickets, stats = mask_dataframe(tickets)
    print(f"ปิดบังเบอร์โทร {stats.phones_masked} จุด · อีเมล {stats.emails_masked} จุด")
    if stats.residual_phones or stats.residual_emails:
        sys.exit(f"หลังปิดบังยังเหลือเบอร์โทร {stats.residual_phones} / อีเมล {stats.residual_emails} "
                 "ไม่ส่งเข้า API ให้ตรวจ src/mask_pii.py ก่อน")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"pred_{safe_model_name(model)}.csv"

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for t in tickets.itertuples(index=False):
            row = {"ticket_id": t.ticket_id, "model": model}
            try:
                labels, resp, latency = classify_one(client, model, system, t.message, effort)
                row.update(labels)
                row.update(
                    stop_reason=resp.stop_reason,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    cache_creation_input_tokens=resp.usage.cache_creation_input_tokens or 0,
                    cache_read_input_tokens=resp.usage.cache_read_input_tokens or 0,
                    latency_s=round(latency, 3),
                    request_id=resp._request_id,
                )
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                    anthropic.NotFoundError) as e:
                # key ผิด / ไม่มีสิทธิ์ / ชื่อโมเดลผิด: ใบต่อไปก็จะพังเหมือนกัน หยุดเลย
                sys.exit(f"{t.ticket_id}: {type(e).__name__}: {e.message}")
            except (anthropic.APIError, RuntimeError, json.JSONDecodeError) as e:
                # rate limit / 5xx ที่ retry แล้วยังไม่ผ่าน, refusal, max_tokens: บันทึกแล้วไปต่อ
                row["error"] = f"{type(e).__name__}: {e}"
            writer.writerow(row)
            f.flush()
            status = row.get("error") or f"{row.get('category')} ({row.get('latency_s')}s)"
            print(f"{t.ticket_id}: {status}")

    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="จัดหมวดหมู่ตั๋วร้องเรียนด้วย Claude API")
    parser.add_argument("model", help="ชื่อโมเดล เช่น claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=5, help="จำนวนตั๋วจากด้านบน (ค่าเริ่มต้น 5)")
    group.add_argument("--all", action="store_true", help="รันทุกใบ")
    group.add_argument("--gold", action="store_true",
                       help="รันเฉพาะ 100 ใบที่อยู่ใน tickets_test_gold.csv (สำหรับ src.evaluate)")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                        help="ระดับ effort (ไม่ใส่ = ค่าเริ่มต้นของโมเดล)")
    args = parser.parse_args(argv)

    limit = None if (args.all or args.gold) else args.limit
    out = run(args.model, limit, args.effort, gold_only=args.gold)
    print(f"เขียนผลแล้ว: {out}")


if __name__ == "__main__":
    main()
