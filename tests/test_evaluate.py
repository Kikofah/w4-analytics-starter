"""test ของ src/evaluate.py ด้วยข้อมูลสมมติ (ไม่เรียก API)"""
import pandas as pd
import pytest

from src import evaluate as ev


@pytest.fixture
def gold():
    return pd.DataFrame({
        "ticket_id": ["T1", "T2", "T3", "T4"],
        "category": ["ส่งช้า", "ขอคืนเงิน", "ชมเชย", "ส่งช้า"],
        "sentiment": ["ลบ", "ลบ", "บวก", "กลาง"],
        "urgency": ["สูง", "สูง", "ปกติ", "ปกติ"],
        "order_id_mentioned": ["BS1", None, None, "BS4"],
        "contains_pii": ["False", "True", "False", "False"],
    })


@pytest.fixture
def pred():
    return pd.DataFrame({
        "ticket_id": ["T1", "T2", "T3", "T4"],
        "category": ["ส่งช้า", "ขอคืนเงิน", "ชมเชย", "ขอคืนเงิน"],  # T4 ผิด
        "sentiment": ["ลบ", "ลบ", "บวก", "กลาง"],
        "urgency": ["สูง", "ปกติ", "ปกติ", "ปกติ"],                 # T2 พลาดสูง
        "order_id_mentioned": ["bs1", "", "", "BS4"],               # ตัวพิมพ์เล็กนับว่าถูก
        "contains_pii": ["False", "True", "False", "False"],
        "model": "claude-haiku-5-5",
        "input_tokens": ["1000"] * 4,
        "output_tokens": ["100"] * 4,
        "cache_creation_input_tokens": ["0"] * 4,
        "cache_read_input_tokens": ["0"] * 4,
        "latency_s": ["1.0", "2.0", "3.0", "4.0"],
        "error": [None] * 4,
    })


def test_accuracy_recall_latency_cost(gold, pred):
    r = ev.evaluate(pred, gold)
    assert r.model == "claude-haiku-5-5"
    assert r.accuracy["category"] == pytest.approx(0.75)
    assert r.accuracy["urgency"] == pytest.approx(0.75)
    assert r.accuracy["order_id_mentioned"] == pytest.approx(1.0)
    assert r.accuracy["contains_pii"] == pytest.approx(1.0)
    assert r.exact_match == pytest.approx(0.5)
    assert r.high_urgency_recall == pytest.approx(0.5)
    assert r.avg_latency_s == pytest.approx(2.5)
    in_price, out_price = ev.PRICES_PER_MTOK["claude-haiku-5-5"]
    per_ticket = (1000 * in_price + 100 * out_price) / 1_000_000
    assert r.cost_per_1000_usd == pytest.approx(per_ticket * 1000)


def test_confusion_matrix(gold, pred):
    cm = ev.evaluate(pred, gold).confusion["category"]
    assert cm.loc["ส่งช้า", "ส่งช้า"] == 1
    assert cm.loc["ส่งช้า", "ขอคืนเงิน"] == 1
    assert cm.values.sum() == 4


def test_error_rows_count_as_wrong(gold, pred):
    pred.loc[0, "error"] = "RateLimitError: ..."
    r = ev.evaluate(pred, gold)
    assert r.n_error == 1
    assert r.accuracy["sentiment"] == pytest.approx(0.75)
    assert r.high_urgency_recall == pytest.approx(0.0)
    assert r.avg_latency_s == pytest.approx(3.0)  # เฉลี่ยเฉพาะใบที่สำเร็จ


def test_no_overlap_raises(gold, pred):
    pred["ticket_id"] = ["X1", "X2", "X3", "X4"]
    with pytest.raises(ValueError):
        ev.evaluate(pred, gold)


def test_comparison_markdown(gold, pred):
    md = ev.build_comparison([ev.evaluate(pred, gold)])
    assert "`claude-haiku-5-5`" in md
    assert "recall urgency=สูง" in md
    assert "Confusion matrix" in md
