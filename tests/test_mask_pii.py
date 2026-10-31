"""test ของ src/mask_pii.py"""
import pandas as pd
import pytest

from src import mask_pii as mp


@pytest.mark.parametrize("phone", [
    "099-555-0329", "0995550329", "099 555 0329", "099.555.0329",
    "+66 99 555 0329", "+66995550329", "02-123-4567", "021234567",
])
def test_masks_thai_phone_formats(phone):
    out = mp.mask_text(f"ติดต่อกลับเบอร์ {phone} นะคะ")
    assert out == "ติดต่อกลับเบอร์ [PHONE] นะคะ"
    assert mp.residual_pii(out) == (0, 0)


def test_masks_phone_glued_to_thai_text():
    assert mp.mask_text("เบอร์0995550329ค่ะ") == "เบอร์[PHONE]ค่ะ"


@pytest.mark.parametrize("email", ["user277@example.com", "a.b+c@mail.co.th", "X_Y@sub.domain.org"])
def test_masks_emails(email):
    out = mp.mask_text(f"อีเมล {email} ค่ะ")
    assert out == "อีเมล [EMAIL] ค่ะ"
    assert mp.residual_pii(out) == (0, 0)


@pytest.mark.parametrize("text", [
    "คำสั่งซื้อ BS2510180010 ค่ะ",           # เลขคำสั่งซื้อต้องอยู่ครบ (ใช้จัดหมวด order_id_mentioned)
    "สั่งไปตั้งแต่วันที่ 2025-11-18",
    "เลขที่ 180/40 ซอย 47",
    "ขวดน้ำ 750 มล. รอมา 13 วัน",
    "ใช้โค้ด BS100 ไม่ได้",
])
def test_does_not_touch_non_pii(text):
    assert mp.mask_text(text) == text
    assert mp.residual_pii(text) == (0, 0)


def test_full_ticket_example():
    msg = "ชื่อ คุณบอส เบอร์ 099-555-0313 อีเมล user277@example.com ที่อยู่ 180/40 ต.ในเมือง"
    assert mp.mask_text(msg) == "ชื่อ คุณบอส เบอร์ [PHONE] อีเมล [EMAIL] ที่อยู่ 180/40 ต.ในเมือง"


def test_residual_detector_catches_unmasked():
    assert mp.residual_pii("โทร 099-555-0329") == (1, 0)
    assert mp.residual_pii("ส่งมาที่ a@b.com") == (0, 1)


def test_real_tickets_no_residual_pii():
    df = pd.read_csv(mp.TICKETS, encoding="utf-8-sig", dtype=str)
    before = df.copy()
    masked, stats = mp.mask_dataframe(df)
    assert df.equals(before)                       # ไม่แก้ข้อมูลต้นฉบับ
    assert stats.phones_masked > 0 and stats.emails_masked > 0
    assert stats.residual_phones == 0
    assert stats.residual_emails == 0
    assert not masked["message"].str.contains("@").any()
    # เลขคำสั่งซื้อยังอยู่ครบทุกตัว
    pattern = r"BS\d{10}"
    assert df["message"].str.count(pattern).sum() == masked["message"].str.count(pattern).sum()
