"""ปิดบังเบอร์โทรและอีเมลในข้อความตั๋วก่อนส่งเข้า Claude API (ตาม policies/04_privacy.md)

ใช้งาน:
    python -m src.mask_pii              # อ่าน data/tickets.csv → เขียน data/tickets_masked.csv และพิมพ์ผลนับ

src/classify.py เรียก mask_text() กับทุกข้อความก่อนส่งเสมอ ไม่ว่าจะอ่านจากไฟล์ไหน
หมายเหตุ: ยังไม่ได้ปิดบังที่อยู่และชื่อ (ไม่มีรูปแบบตายตัวพอสำหรับ regex)
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TICKETS = ROOT / "data" / "tickets.csv"
MASKED = ROOT / "data" / "tickets_masked.csv"

PHONE_TOKEN = "[PHONE]"
EMAIL_TOKEN = "[EMAIL]"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

# เบอร์ไทย: ขึ้นต้น 0 หรือ +66 ตามด้วยตัวเลขอีก 8–9 หลัก คั่นด้วย - . ช่องว่าง หรือไม่คั่นก็ได้
#   มือถือ 099-555-0329 / 0995550329 / 099 555 0329 / +66 99 555 0329
#   บ้าน   02-123-4567 / 021234567
# ข้างหน้าต้องไม่ใช่ตัวอักษรหรือตัวเลขภาษาอังกฤษ จึงไม่ไปโดนเลขคำสั่งซื้อ BS2510180010
PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9+])"
    r"(?:\+66[\s.-]?|0)"
    r"\(?\d\)?(?:[\s.-]?\d){7,8}"
    r"(?![0-9])"
)

# ตัวตรวจหลังปิดบัง: กว้างกว่าตัวปิดบังโดยตั้งใจ เพื่อจับสิ่งที่หลุดรอด
#   - มี @ ที่ตามด้วยโดเมน
#   - ตัวเลข 9–10 หลักติดกันหลังตัดตัวคั่นออก (ยกเว้นเลขคำสั่งซื้อ BS… และวันที่ YYYY-MM-DD)
RESIDUAL_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
RESIDUAL_PHONE_RE = re.compile(r"(?<!\d)(?:\+?66|0)\d{8,9}(?!\d)")
_SAFE_RE = re.compile(r"BS\d+|\d{4}-\d{2}-\d{2}")


@dataclass
class MaskStats:
    rows: int
    rows_with_pii: int
    phones_masked: int
    emails_masked: int
    residual_phones: int
    residual_emails: int


def mask_text(text: str) -> str:
    """แทนอีเมลด้วย [EMAIL] และเบอร์โทรด้วย [PHONE] (ทำอีเมลก่อน กันตัวเลขในอีเมลถูกมองเป็นเบอร์)"""
    if not isinstance(text, str):
        return text
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)
    return PHONE_RE.sub(PHONE_TOKEN, text)


def residual_pii(text: str) -> tuple[int, int]:
    """นับรูปแบบเบอร์โทร/อีเมลที่ยังเหลือ คืน (จำนวนเบอร์, จำนวนอีเมล)"""
    if not isinstance(text, str):
        return 0, 0
    emails = len(RESIDUAL_EMAIL_RE.findall(text))
    digits_only = re.sub(r"(?<=\d)[\s.()-]+(?=\d)", "", _SAFE_RE.sub(" ", text))
    phones = len(RESIDUAL_PHONE_RE.findall(digits_only))
    return phones, emails


def mask_dataframe(df: pd.DataFrame, column: str = "message") -> tuple[pd.DataFrame, MaskStats]:
    """คืน DataFrame ใหม่ที่ปิดบังแล้ว (ไม่แก้ df เดิม) พร้อมผลนับก่อน/หลัง"""
    original = df[column].fillna("")
    phones = original.map(lambda s: len(PHONE_RE.findall(EMAIL_RE.sub(EMAIL_TOKEN, s))))
    emails = original.map(lambda s: len(EMAIL_RE.findall(s)))

    out = df.copy()
    out[column] = df[column].map(mask_text)
    residual = out[column].map(residual_pii)

    stats = MaskStats(
        rows=len(df),
        rows_with_pii=int(((phones + emails) > 0).sum()),
        phones_masked=int(phones.sum()),
        emails_masked=int(emails.sum()),
        residual_phones=int(residual.map(lambda r: r[0]).sum()),
        residual_emails=int(residual.map(lambda r: r[1]).sum()),
    )
    return out, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ปิดบังเบอร์โทรและอีเมลใน tickets.csv")
    parser.add_argument("--input", type=Path, default=TICKETS)
    parser.add_argument("-o", "--output", type=Path, default=MASKED)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str)
    masked, s = mask_dataframe(df)
    masked.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"ตั๋วทั้งหมด {s.rows} ใบ · มีเบอร์/อีเมล {s.rows_with_pii} ใบ")
    print(f"ปิดบังเบอร์โทร {s.phones_masked} จุด · อีเมล {s.emails_masked} จุด")
    print(f"หลังปิดบัง เหลือรูปแบบเบอร์โทร {s.residual_phones} · อีเมล {s.residual_emails}")
    print(f"เขียนไฟล์แล้ว: {args.output}")
    if s.residual_phones or s.residual_emails:
        sys.exit("ยังเหลือข้อมูลที่ดูเหมือนเบอร์โทรหรืออีเมล ตรวจ regex ก่อนส่งเข้า API")


if __name__ == "__main__":
    main()
