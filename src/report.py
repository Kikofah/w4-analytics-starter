"""สร้างรายงานยอดขายรายเดือน

ใช้งาน: python -m src.report 2026-03
ผลลัพธ์: reports/monthly_report.md
"""
import argparse
import math
import re
from pathlib import Path

import pandas as pd

from src import metrics as m

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def previous_month(month: str) -> str:
    """'2026-03' -> '2026-02', '2026-01' -> '2025-12'"""
    year, mon = int(month[:4]), int(month[5:7])
    if mon == 1:
        return f"{year - 1}-12"
    return f"{year}-{mon - 1:02d}"


def _baht(x: float) -> str:
    return f"{x:,.0f} บาท"


def _pct(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:.2f}%"


def build_report(df: pd.DataFrame, month: str) -> str:
    """คืนเนื้อหารายงาน Markdown ของเดือนที่ระบุ ('YYYY-MM')"""
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"รูปแบบเดือนต้องเป็น YYYY-MM ได้รับ: {month!r}")

    cur = df[df["month"] == month]
    if cur.empty:
        raise ValueError(f"ไม่พบข้อมูลของเดือน {month}")

    prev_month = previous_month(month)
    prev = df[df["month"] == prev_month]

    sales = m.net_sales(cur)
    lines = [
        f"# รายงานยอดขายประจำเดือน {month}",
        "",
        "## ภาพรวม",
        "",
        "| ตัวชี้วัด | ค่า |",
        "|---|---:|",
        f"| ยอดขายสุทธิ | {_baht(sales)} |",
    ]

    if prev.empty:
        lines.append(f"| เทียบเดือนก่อน ({prev_month}) | ไม่มีข้อมูล |")
    else:
        prev_sales = m.net_sales(prev)
        diff = sales - prev_sales
        change = f" ({diff / prev_sales * 100:+.2f}%)" if prev_sales else ""
        lines.append(f"| ยอดขายสุทธิเดือนก่อน ({prev_month}) | {_baht(prev_sales)} |")
        lines.append(f"| เปลี่ยนแปลงจากเดือนก่อน | {diff:+,.0f} บาท{change} |")

    lines += [
        f"| อัตรายกเลิก | {_pct(m.cancel_rate(cur))} |",
        f"| อัตราคืนสินค้า | {_pct(m.return_rate(cur))} |",
        "",
        "## 5 สินค้าขายดี (ตามยอดขายสุทธิ)",
        "",
        "| อันดับ | รหัส | สินค้า | จำนวนชิ้น | ยอดขายสุทธิ |",
        "|---:|---|---|---:|---:|",
    ]
    for i, row in m.top_products(cur, 5).iterrows():
        lines.append(f"| {i + 1} | {row['product_id']} | {row['product_name']} | "
                     f"{int(row['qty']):,} | {_baht(row['net'])} |")

    lines += ["", "_นิยามตัวชี้วัดตาม CLAUDE.md · ยอดขายนับเฉพาะคำสั่งซื้อที่จัดส่งสำเร็จ ไม่รวมค่าส่ง_", ""]
    return "\n".join(lines)


def write_report(month: str, out_path: Path | str = REPORTS / "monthly_report.md",
                 df: pd.DataFrame | None = None) -> Path:
    """สร้างรายงานแล้วเขียนลงไฟล์ คืน path ของไฟล์"""
    if df is None:
        df = m.load_orders()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(df, month), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="สร้างรายงานยอดขายรายเดือน")
    parser.add_argument("month", help="เดือนในรูปแบบ YYYY-MM เช่น 2026-03")
    parser.add_argument("-o", "--output", default=REPORTS / "monthly_report.md",
                        help="ไฟล์ผลลัพธ์ (ค่าเริ่มต้น reports/monthly_report.md)")
    args = parser.parse_args(argv)
    path = write_report(args.month, args.output)
    print(f"เขียนรายงานแล้ว: {path}")


if __name__ == "__main__":
    main()
