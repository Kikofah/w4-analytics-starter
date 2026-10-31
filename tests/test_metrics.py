"""ตัวเลขที่ต้องได้ (สัญญาของโปรเจกต์) ห้ามแก้ไฟล์นี้"""
import pytest

from src import metrics as m


@pytest.fixture(scope="module")
def df():
    return m.load_orders()


@pytest.fixture(scope="module")
def sample():
    return m.load_orders(m.DATA / "orders_sample_80.csv")


def test_load(df):
    assert len(df) == 17529
    assert {"net", "gp", "month", "date"} <= set(df.columns)


def test_net_sales(df):
    assert m.net_sales(df) == pytest.approx(14_107_737.05, abs=1)


def test_net_sales_sample_not_hardcoded(sample):
    # ตัวอย่าง 80 แถว: 72 แถวจัดส่งสำเร็จ
    assert m.net_sales(sample) == pytest.approx(72_781, abs=1)


def test_gross_profit(df):
    assert m.gross_profit(df) == pytest.approx(6_201_497, abs=2)
    assert m.gross_profit(df) / m.net_sales(df) == pytest.approx(0.440, abs=0.001)


def test_aov(df):
    assert m.aov(df) == pytest.approx(868.22, abs=0.01)


def test_cancel_rate(df):
    assert m.cancel_rate(df) == pytest.approx(4.51, abs=0.01)


def test_monthly(df):
    s = m.monthly_sales(df)
    assert len(s) == 12
    assert list(s.index) == sorted(s.index)
    assert s["2026-03"] == pytest.approx(1_147_154.90, abs=1)
    assert m.best_month(df) == "2025-11"


def test_channel_share(df):
    assert m.channel_share(df, "แอป", "2026-07", "2026-09") == pytest.approx(44.85, abs=0.02)
    assert m.channel_share(df, "แอป", "2025-10", "2025-12") == pytest.approx(32.0, abs=0.05)


def test_rating_by_ship_bucket(df):
    r = m.rating_by_ship_bucket(df)
    assert r["1-2"] == pytest.approx(4.37, abs=0.01)
    assert r["3-4"] == pytest.approx(4.31, abs=0.01)
    assert r["5-6"] == pytest.approx(4.07, abs=0.01)
    assert r["7+"] == pytest.approx(3.53, abs=0.01)


def test_top_province(df):
    assert m.top_province(df, "ภาคเหนือ") == "เชียงใหม่"


def test_flood_delay(df):
    d = m.avg_ship_days(df, region="ภาคใต้", carrier="ไปไว Express",
                        start_date="2025-11-15", end_date="2025-12-25")
    assert d == pytest.approx(10.51, abs=0.02)


def test_fryer_returns(df):
    assert m.return_rate(df, "P001", "2026-03-01", "2026-04-30") == pytest.approx(24.74, abs=0.02)
