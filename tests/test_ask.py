"""test ของ src/ask.py และ src/run_eval.py (ไม่เรียก API จริง)"""
import json
from types import SimpleNamespace

import pytest

from src import ask as a
from src import metrics as m
from src.run_eval import grade, short_answer


@pytest.fixture(scope="module")
def df():
    return m.load_orders()


# ---------- เครื่องมือพอสำหรับตอบทุกข้อใน eval ----------

def run_tool(df, name, **tool_input):
    return a.execute_tool(df, name, tool_input)


def test_tools_cover_eval_answers(df):
    s = run_tool(df, "net_sales")
    assert s["net_sales"] == pytest.approx(14_107_737.05, abs=0.01)           # Q01
    assert s["cancel_rate_pct"] == pytest.approx(4.51, abs=0.01)              # Q04
    assert s["aov"] == pytest.approx(868.22, abs=0.01)                        # Q08
    assert s["unique_customers"] == 5322                                      # Q13

    monthly = run_tool(df, "monthly_sales")["monthly_net_sales"]
    assert max(monthly, key=monthly.get) == "2025-11"                         # Q02
    assert monthly["2026-03"] == pytest.approx(1_147_154.90, abs=0.01)        # Q03

    top = lambda **kw: run_tool(df, "top_n", n=1, **kw)["items"][0]
    assert top(dimension="region")["region"] == "กรุงเทพฯและปริมณฑล"          # Q05
    assert top(dimension="category", metric="gp_margin_pct")["category"] == "ต้นไม้และสวน"  # Q06
    assert top(dimension="product_name")["product_name"] == "หม้อทอดไร้น้ำมัน 5 ลิตร"    # Q07
    assert top(dimension="date")["date"] == "2025-11-11"                     # Q12
    assert top(dimension="province", filters={"region": "ภาคเหนือ"})["province"] == "เชียงใหม่"  # Q14

    ch = run_tool(df, "top_n", dimension="channel", n=4,
                  filters={"start_month": "2026-07", "end_month": "2026-09"})["items"]
    app = next(i for i in ch if i["channel"] == "แอป")
    assert app["share_pct"] == pytest.approx(44.85, abs=0.01)                # Q09

    d = run_tool(df, "avg_ship_days", filters={"region": "ภาคใต้", "carrier": "ไปไว Express",
                                               "start_date": "2025-11-15", "end_date": "2025-12-25"})
    assert d["avg_ship_days"] == pytest.approx(10.51, abs=0.01)              # Q10

    r = run_tool(df, "rate", metric="return_rate",
                 filters={"product_name": "หม้อทอดไร้น้ำมัน 5 ลิตร",
                          "start_date": "2026-03-01", "end_date": "2026-04-30"})
    assert r["value_pct"] == pytest.approx(24.74, abs=0.01)                  # Q11


def test_tools_are_read_only(df):
    before = df.copy()
    for t in a.TOOLS:
        name = t["name"]
        assert not any(w in name for w in ("delete", "update", "write", "insert", "drop"))
    run_tool(df, "net_sales", filters={"region": "ภาคใต้"})
    run_tool(df, "top_n", dimension="province")
    assert df.equals(before)


def test_unknown_filter_rejected(df):
    with pytest.raises(ValueError):
        run_tool(df, "net_sales", filters={"status": "ยกเลิก"})


def test_tool_schemas_valid():
    for t in a.TOOLS:
        assert t["input_schema"]["type"] == "object"
        assert t["input_schema"]["additionalProperties"] is False
    json.dumps(a.TOOLS, ensure_ascii=False)


# ---------- วนรอบ tool use ด้วย client จำลอง ----------

class FakeClient:
    """จำลอง client: รอบแรกขอเรียก net_sales รอบสองตอบเสร็จ"""

    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        if len(self.calls) == 1:
            block = SimpleNamespace(type="tool_use", id="tu_1", name="net_sales", input={})
            return SimpleNamespace(stop_reason="tool_use", content=[block], usage=usage)
        result = kwargs["messages"][-1]["content"][0]
        value = json.loads(result["content"])["net_sales"]
        text = SimpleNamespace(type="text", text=f"ยอดขาย {value}\nเครื่องมือที่ใช้: net_sales\nคำตอบสั้น: {value}")
        return SimpleNamespace(stop_reason="end_turn", content=[text], usage=usage)


def test_ask_loop(df):
    client = FakeClient()
    r = a.ask("ยอดขายทั้งปี", df, client, model="claude-opus-5-5")
    assert r.turns == 2
    assert [t["name"] for t in r.tools_used] == ["net_sales"]
    assert r.input_tokens == 200
    ok, got = grade("number", "14107737.05", 0.005, r.answer, r.tools_used)
    assert ok and got.startswith("14107737")
    # ลำดับข้อความ: user → assistant(tool_use) → user(tool_result) → assistant(คำตอบ)
    msgs = client.calls[1]["messages"]
    assert [x["role"] for x in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "tu_1"
    # โมเดลที่รองรับต้องเปิด server-side fallback
    assert client.calls[0]["fallbacks"] == "default"


# ---------- การให้คะแนน ----------

def test_grade_number_tolerance():
    assert grade("number", "4.51", 0.02, "คำตอบสั้น: 4.5", [])[0]
    assert not grade("number", "4.51", 0.02, "คำตอบสั้น: 4.0", [])[0]
    assert grade("number", "14107737.05", 0.005, "คำตอบสั้น: 14,107,737 บาท", [])[0]
    assert not grade("number", "868.22", 0.005, "ไม่มีบรรทัดคำตอบสั้น 868.22", [])[0]


def test_grade_text_and_refuse():
    assert grade("text", "2025-11", 0, "...\nคำตอบสั้น: 2025-11", [])[0]
    assert grade("text", "เชียงใหม่", 0, "คำตอบสั้น: จังหวัดเชียงใหม่", [])[0]
    assert not grade("text", "เชียงใหม่", 0, "คำตอบสั้น: ลำพูน", [])[0]
    assert grade("refuse", "", 0, "ขออภัย ทำไม่ได้\nคำตอบสั้น: ปฏิเสธ", [])[0]
    assert not grade("refuse", "", 0, "คำตอบสั้น: ปฏิเสธ", [{"name": "net_sales"}])[0]


def test_short_answer_takes_last_line():
    assert short_answer("คำตอบสั้น: 1\nแก้ไข\nคำตอบสั้น: **2**") == "2"
