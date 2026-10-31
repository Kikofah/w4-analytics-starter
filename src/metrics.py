"""ฟังก์ชันคำนวณตัวชี้วัดของบ้านสบาย ช็อป

ทุกฟังก์ชันรับ DataFrame ที่ได้จาก load_orders() เป็นพารามิเตอร์แรก
นิยามตัวชี้วัดอยู่ใน CLAUDE.md
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

DELIVERED = "จัดส่งสำเร็จ"
CANCELLED = "ยกเลิก"
RETURNED = "คืนสินค้า"


def load_orders(path: Path | str = DATA / "orders.csv") -> pd.DataFrame:
    """อ่าน orders.csv แล้วเพิ่มคอลัมน์ที่ใช้บ่อย: net, unit_cost, gp, month, date

    net และ gp คำนวณทุกแถว (ไม่กรองสถานะ) ฟังก์ชันที่ใช้ต้องกรอง status เอง
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    products = pd.read_csv(DATA / "products.csv", encoding="utf-8-sig")
    df = df.merge(products[["product_id", "unit_cost"]], on="product_id", how="left")
    df["net"] = df["qty"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
    df["gp"] = df["net"] - df["qty"] * df["unit_cost"]
    df["month"] = df["order_datetime"].str[:7]
    df["date"] = df["order_datetime"].str[:10]
    return df


def _delivered(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == DELIVERED]


def _filter_dates(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]
    return df


def _status_rate(df: pd.DataFrame, status: str) -> float:
    if len(df) == 0:
        return float("nan")
    return float((df["status"] == status).sum() / len(df) * 100)


def net_sales(df: pd.DataFrame) -> float:
    """ยอดขายสุทธิรวม (บาท)"""
    return float(_delivered(df)["net"].sum())


def gross_profit(df: pd.DataFrame) -> float:
    """กำไรขั้นต้นรวม (บาท)"""
    return float(_delivered(df)["gp"].sum())


def aov(df: pd.DataFrame) -> float:
    """ยอดขายสุทธิเฉลี่ยต่อคำสั่งซื้อที่จัดส่งสำเร็จ (บาท)"""
    d = _delivered(df)
    return float(d["net"].sum() / d["order_id"].nunique())


def cancel_rate(df: pd.DataFrame) -> float:
    """อัตรายกเลิกของคำสั่งซื้อทั้งหมดใน df (เปอร์เซ็นต์ 0–100)"""
    return _status_rate(df, CANCELLED)


def monthly_sales(df: pd.DataFrame) -> pd.Series:
    """ยอดขายสุทธิรายเดือน index = 'YYYY-MM' เรียงตามเวลา"""
    return _delivered(df).groupby("month")["net"].sum().sort_index()


def best_month(df: pd.DataFrame) -> str:
    """เดือนที่ยอดขายสุทธิสูงสุด ('YYYY-MM')"""
    return str(monthly_sales(df).idxmax())


def channel_share(df: pd.DataFrame, channel: str, start_month: str, end_month: str) -> float:
    """สัดส่วนยอดขายสุทธิของช่องทางในช่วงเดือน start_month..end_month (รวมทั้งสองเดือน) เป็นเปอร์เซ็นต์"""
    d = _delivered(df)
    d = d[d["month"].between(start_month, end_month)]
    total = d["net"].sum()
    return float(d.loc[d["channel"] == channel, "net"].sum() / total * 100)


def rating_by_ship_bucket(df: pd.DataFrame) -> dict[str, float]:
    """คะแนนรีวิวเฉลี่ยตามระยะส่ง คืน dict คีย์ '1-2', '3-4', '5-6', '7+'

    ใช้ทุกแถวที่มีทั้ง rating และ ship_days (รวมคำสั่งซื้อที่คืนสินค้า)
    """
    d = df.dropna(subset=["rating", "ship_days"])
    labels = ["1-2", "3-4", "5-6", "7+"]
    buckets = pd.cut(d["ship_days"], bins=[0, 2, 4, 6, float("inf")], labels=labels)
    means = d.groupby(buckets, observed=False)["rating"].mean()
    return {label: float(means[label]) for label in labels}


def top_province(df: pd.DataFrame, region: str) -> str:
    """จังหวัดที่มียอดขายสุทธิสูงสุดในภูมิภาคที่ระบุ"""
    d = _delivered(df)
    d = d[d["region"] == region]
    return str(d.groupby("province")["net"].sum().idxmax())


def avg_ship_days(df: pd.DataFrame, region: str | None = None, carrier: str | None = None,
                  start_date: str | None = None, end_date: str | None = None) -> float:
    """ระยะส่งเฉลี่ย (วัน) ตามตัวกรองที่ให้มา วันที่เป็น 'YYYY-MM-DD' และรวมทั้งสองวัน

    เฉลี่ยจากทุกแถวที่มี ship_days (คำสั่งซื้อที่ยกเลิกไม่มีค่า จึงถูกตัดออกเอง)
    """
    d = df
    if region is not None:
        d = d[d["region"] == region]
    if carrier is not None:
        d = d[d["carrier"] == carrier]
    d = _filter_dates(d, start_date, end_date)
    return float(d["ship_days"].dropna().mean())


def return_rate(df: pd.DataFrame, product_id: str | None = None,
                start_date: str | None = None, end_date: str | None = None) -> float:
    """อัตราคืนสินค้า (เปอร์เซ็นต์) ของสินค้าและช่วงวันที่ที่ระบุ

    ตัวหาร = คำสั่งซื้อทั้งหมดในกลุ่ม (รวมที่ยกเลิก) ตามนิยามใน CLAUDE.md
    """
    d = df
    if product_id is not None:
        d = d[d["product_id"] == product_id]
    d = _filter_dates(d, start_date, end_date)
    return _status_rate(d, RETURNED)


def top_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """สินค้าขายดี n อันดับแรกตามยอดขายสุทธิ (เฉพาะจัดส่งสำเร็จ)

    คืน DataFrame คอลัมน์ product_id, product_name, qty, net เรียงจากมากไปน้อย
    """
    d = _delivered(df)
    out = (d.groupby(["product_id", "product_name"], as_index=False)[["qty", "net"]].sum()
             .sort_values("net", ascending=False)
             .head(n)
             .reset_index(drop=True))
    return out


# ---------- ฟังก์ชันทั่วไปสำหรับ src/ask.py (อ่านอย่างเดียว) ----------

FILTER_KEYS = ("start_date", "end_date", "start_month", "end_month", "region", "province",
               "channel", "category", "product_id", "product_name", "carrier")

DIMENSIONS = ("month", "date", "region", "province", "channel", "category",
              "product_name", "carrier", "payment_method")

TOP_N_METRICS = ("net_sales", "gross_profit", "gp_margin_pct", "orders", "qty")


def filter_orders(df: pd.DataFrame, start_date: str | None = None, end_date: str | None = None,
                  start_month: str | None = None, end_month: str | None = None,
                  region: str | None = None, province: str | None = None,
                  channel: str | None = None, category: str | None = None,
                  product_id: str | None = None, product_name: str | None = None,
                  carrier: str | None = None) -> pd.DataFrame:
    """คืนแถวที่ผ่านตัวกรอง (ไม่แก้ df) วันที่/เดือนรวมทั้งสองฝั่ง product_name จับแบบมีคำนี้อยู่"""
    d = _filter_dates(df, start_date, end_date)
    if start_month is not None:
        d = d[d["month"] >= start_month]
    if end_month is not None:
        d = d[d["month"] <= end_month]
    for col, val in (("region", region), ("province", province), ("channel", channel),
                     ("category", category), ("product_id", product_id), ("carrier", carrier)):
        if val is not None:
            d = d[d[col] == val]
    if product_name is not None:
        d = d[d["product_name"].str.contains(product_name, regex=False)]
    return d


def summary(df: pd.DataFrame) -> dict[str, float | int]:
    """ตัวชี้วัดหลักของแถวใน df: ยอดขาย กำไร AOV จำนวนคำสั่งซื้อ ลูกค้าไม่ซ้ำ อัตรายกเลิก/คืน"""
    d = _delivered(df)
    ns = float(d["net"].sum())
    gp = float(d["gp"].sum())
    n_delivered = int(d["order_id"].nunique())
    return {
        "rows_matched": int(len(df)),
        "net_sales": ns,
        "gross_profit": gp,
        "gp_margin_pct": gp / ns * 100 if ns else float("nan"),
        "delivered_orders": n_delivered,
        "aov": ns / n_delivered if n_delivered else float("nan"),
        "total_orders": int(df["order_id"].nunique()),
        "unique_customers": int(df["customer_id"].nunique()),
        "cancel_rate_pct": cancel_rate(df),
        "return_rate_pct": _status_rate(df, RETURNED),
    }


def top_n(df: pd.DataFrame, dimension: str, n: int = 5, metric: str = "net_sales") -> pd.DataFrame:
    """อันดับ n แรกของ dimension เรียงตาม metric (เฉพาะจัดส่งสำเร็จ)

    คืนคอลัมน์ dimension, value และ share_pct (สัดส่วนต่อทั้งหมด สำหรับ metric ที่บวกกันได้)
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension ต้องเป็นหนึ่งใน {DIMENSIONS}")
    if metric not in TOP_N_METRICS:
        raise ValueError(f"metric ต้องเป็นหนึ่งใน {TOP_N_METRICS}")
    d = _delivered(df)
    g = d.groupby(dimension)
    if metric == "net_sales":
        value = g["net"].sum()
    elif metric == "gross_profit":
        value = g["gp"].sum()
    elif metric == "gp_margin_pct":
        value = g["gp"].sum() / g["net"].sum() * 100
    elif metric == "orders":
        value = g["order_id"].nunique()
    else:
        value = g["qty"].sum()
    out = value.rename("value").sort_values(ascending=False).reset_index()
    if metric != "gp_margin_pct":
        total = value.sum()
        out["share_pct"] = out["value"] / total * 100 if total else float("nan")
    return out.head(n)
