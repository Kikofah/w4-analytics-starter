"""รายงานยอดขายรายเดือน 1 หน้า · บ้านสบาย ช็อป

ดัดแปลงจาก src/report.py ให้ทำงานได้ด้วยตัวเอง (ไม่ต้องมี src/metrics.py) ใช้แค่ pandas

ใช้งาน:
    python scripts/report.py orders.csv 2026-03
    python scripts/report.py orders.csv 2026-03 --products products.csv -o monthly_report.md
    python scripts/report.py orders.csv --check            # ตรวจข้อมูลอย่างเดียว ไม่ทำรายงาน

ขั้นตอน:
    1. ตรวจข้อมูล: จำนวนแถว จำนวนแต่ละ status คอลัมน์ที่ต้องมี ช่วงเดือน
    2. คำนวณตามนิยาม: ยอดขายสุทธิ = qty × unit_price × (1 − discount_pct/100)
       เฉพาะ status == "จัดส่งสำเร็จ" ไม่รวม shipping_fee
    3. เขียนรายงาน Markdown 1 หน้า
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DELIVERED = "จัดส่งสำเร็จ"
CANCELLED = "ยกเลิก"
RETURNED = "คืนสินค้า"
KNOWN_STATUSES = (DELIVERED, CANCELLED, RETURNED)

REQUIRED_COLUMNS = ("order_id", "order_datetime", "product_id", "product_name",
                    "qty", "unit_price", "discount_pct", "status")


# ---------- โหลดและตรวจข้อมูล ----------

def load_orders(path: Path | str, products_path: Path | str | None = None) -> pd.DataFrame:
    """อ่าน orders.csv (UTF-8 มี BOM) แล้วเพิ่มคอลัมน์ net, month, date และ gp ถ้ามี products.csv"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ไฟล์ขาดคอลัมน์ที่ต้องใช้: {missing}")
    df["net"] = df["qty"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
    df["month"] = df["order_datetime"].astype(str).str[:7]
    df["date"] = df["order_datetime"].astype(str).str[:10]
    if products_path is not None:
        products = pd.read_csv(products_path, encoding="utf-8-sig")
        df = df.merge(products[["product_id", "unit_cost"]], on="product_id", how="left")
        df["gp"] = df["net"] - df["qty"] * df["unit_cost"]
    return df


def check_data(df: pd.DataFrame) -> dict:
    """ข้อมูลสำหรับให้ผู้ใช้ตรวจก่อนเชื่อตัวเลข"""
    status_counts = df["status"].value_counts().to_dict()
    return {
        "rows": int(len(df)),
        "unique_orders": int(df["order_id"].nunique()),
        "status_counts": status_counts,
        "unknown_statuses": sorted(set(status_counts) - set(KNOWN_STATUSES)),
        "months": sorted(df["month"].unique()),
        "net_all_statuses": float(df["net"].sum()),
        "net_delivered": float(df.loc[df["status"] == DELIVERED, "net"].sum()),
    }


def format_check(c: dict) -> str:
    lines = [
        f"จำนวนแถว: {c['rows']:,} · คำสั่งซื้อไม่ซ้ำ: {c['unique_orders']:,}",
        "จำนวนแต่ละ status: " + " · ".join(f"{k} {v:,}" for k, v in c["status_counts"].items()),
        f"ช่วงข้อมูล: {c['months'][0]} ถึง {c['months'][-1]} ({len(c['months'])} เดือน)",
        f"ยอดรวมทุกสถานะ (ผิดนิยาม): {c['net_all_statuses']:,.2f} บาท",
        f"ยอดขายสุทธิเฉพาะจัดส่งสำเร็จ: {c['net_delivered']:,.2f} บาท",
    ]
    if c["unknown_statuses"]:
        lines.append(f"⚠️ พบ status ที่ไม่รู้จัก: {c['unknown_statuses']}")
    if c["rows"] != c["unique_orders"]:
        lines.append("⚠️ order_id ซ้ำ: 1 แถวไม่เท่ากับ 1 คำสั่งซื้อ ตรวจก่อนตีความ AOV และอัตรา")
    return "\n".join(lines)


# ---------- ตัวชี้วัด ----------

def previous_month(month: str) -> str:
    """'2026-03' -> '2026-02', '2026-01' -> '2025-12'"""
    year, mon = int(month[:4]), int(month[5:7])
    return f"{year - 1}-12" if mon == 1 else f"{year}-{mon - 1:02d}"


def net_sales(df: pd.DataFrame) -> float:
    return float(df.loc[df["status"] == DELIVERED, "net"].sum())


def status_rate(df: pd.DataFrame, status: str) -> float:
    return float((df["status"] == status).mean() * 100) if len(df) else float("nan")


def top_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    d = df[df["status"] == DELIVERED]
    return (d.groupby(["product_id", "product_name"], as_index=False)[["qty", "net"]].sum()
              .sort_values("net", ascending=False).head(n).reset_index(drop=True))


# ---------- รายงาน ----------

def _baht(x: float) -> str:
    return f"{x:,.0f} บาท"


def _pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2f}%"


def build_report(df: pd.DataFrame, month: str) -> str:
    """รายงาน Markdown 1 หน้าของเดือนที่ระบุ ('YYYY-MM')"""
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"รูปแบบเดือนต้องเป็น YYYY-MM ได้รับ: {month!r}")
    cur = df[df["month"] == month]
    if cur.empty:
        raise ValueError(f"ไม่พบข้อมูลของเดือน {month} (มีข้อมูล {df['month'].min()} ถึง {df['month'].max()})")

    prev_month = previous_month(month)
    prev = df[df["month"] == prev_month]
    sales = net_sales(cur)
    delivered = cur[cur["status"] == DELIVERED]
    n_delivered = delivered["order_id"].nunique()
    counts = cur["status"].value_counts()

    lines = [
        f"# รายงานยอดขายประจำเดือน {month} · บ้านสบาย ช็อป",
        "",
        f"ข้อมูลเดือนนี้ {len(cur):,} คำสั่งซื้อ · "
        + " · ".join(f"{s} {int(counts.get(s, 0)):,}" for s in KNOWN_STATUSES),
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
        prev_sales = net_sales(prev)
        diff = sales - prev_sales
        change = f" ({diff / prev_sales * 100:+.2f}%)" if prev_sales else ""
        lines.append(f"| ยอดขายสุทธิเดือนก่อน ({prev_month}) | {_baht(prev_sales)} |")
        lines.append(f"| เปลี่ยนแปลงจากเดือนก่อน | {diff:+,.0f} บาท{change} |")
    if "gp" in cur:
        gp = float(delivered["gp"].sum())
        lines.append(f"| กำไรขั้นต้น | {_baht(gp)} ({gp / sales * 100:.1f}%) |" if sales else "| กำไรขั้นต้น | - |")
    lines += [
        f"| AOV (ต่อคำสั่งซื้อจัดส่งสำเร็จ) | {sales / n_delivered:,.2f} บาท |" if n_delivered else "| AOV | - |",
        f"| อัตรายกเลิก | {_pct(status_rate(cur, CANCELLED))} |",
        f"| อัตราคืนสินค้า | {_pct(status_rate(cur, RETURNED))} |",
        "",
        "## 5 สินค้าขายดี (ตามยอดขายสุทธิ)",
        "",
        "| อันดับ | รหัส | สินค้า | จำนวนชิ้น | ยอดขายสุทธิ |",
        "|---:|---|---|---:|---:|",
    ]
    for i, row in top_products(cur).iterrows():
        lines.append(f"| {i + 1} | {row['product_id']} | {row['product_name']} | "
                     f"{int(row['qty']):,} | {_baht(row['net'])} |")
    lines += [
        "",
        "## ข้อสังเกต",
        "",
        "<!-- 2–3 ข้อ อิงตัวเลขด้านบนเท่านั้น -->",
        "",
        "---",
        "_ยอดขายสุทธิ = qty × unit_price × (1 − discount_pct/100) เฉพาะคำสั่งซื้อจัดส่งสำเร็จ ไม่รวมค่าส่ง · "
        "อัตรายกเลิก/คืน = จำนวนคำสั่งซื้อสถานะนั้น ÷ คำสั่งซื้อทั้งหมดของเดือน × 100_",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="รายงานยอดขายรายเดือน บ้านสบาย ช็อป")
    parser.add_argument("orders", type=Path, help="path ของ orders.csv")
    parser.add_argument("month", nargs="?", help="เดือน YYYY-MM เช่น 2026-03")
    parser.add_argument("--products", type=Path, help="products.csv (ถ้ามี จะเพิ่มกำไรขั้นต้น)")
    parser.add_argument("-o", "--output", type=Path, default=Path("monthly_report.md"))
    parser.add_argument("--check", action="store_true", help="ตรวจข้อมูลอย่างเดียว")
    args = parser.parse_args(argv)

    try:
        df = load_orders(args.orders, args.products)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"อ่านข้อมูลไม่ได้: {e}")

    print("== ตรวจข้อมูล ==")
    print(format_check(check_data(df)))
    if args.check:
        return
    if not args.month:
        parser.error("ต้องระบุเดือน YYYY-MM (หรือใช้ --check)")

    try:
        report = build_report(df, args.month)
    except ValueError as e:
        sys.exit(str(e))
    args.output.write_text(report, encoding="utf-8")
    print(f"\nเขียนรายงานแล้ว: {args.output}")


if __name__ == "__main__":
    main()
