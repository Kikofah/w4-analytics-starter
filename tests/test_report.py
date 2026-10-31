"""test ของ src/report.py"""
import pytest

from src import metrics as m
from src import report


@pytest.fixture(scope="module")
def df():
    return m.load_orders()


def test_report_2026_03_net_sales(df, tmp_path):
    path = report.write_report("2026-03", tmp_path / "monthly_report.md", df=df)
    text = path.read_text(encoding="utf-8")
    assert "2026-03" in text
    assert "| ยอดขายสุทธิ | 1,147,155 บาท |" in text


def test_report_sections(df):
    text = report.build_report(df, "2026-03")
    for heading in ["เดือนก่อน (2026-02)", "เปลี่ยนแปลงจากเดือนก่อน",
                    "อัตรายกเลิก", "อัตราคืนสินค้า", "5 สินค้าขายดี"]:
        assert heading in text
    # ตารางสินค้าขายดีต้องมี 5 แถว
    assert sum(1 for line in text.splitlines() if line.startswith("| ") and " | P" in line) == 5


def test_previous_month():
    assert report.previous_month("2026-03") == "2026-02"
    assert report.previous_month("2026-01") == "2025-12"


def test_first_month_has_no_previous(df):
    text = report.build_report(df, "2025-10")
    assert "ไม่มีข้อมูล" in text


def test_report_rejects_bad_month(df):
    with pytest.raises(ValueError):
        report.build_report(df, "2026-13")
