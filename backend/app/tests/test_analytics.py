from app.services.analytics import parse_tx_date, summarize_transactions, summary_markdown


def test_parse_tx_date_iso_and_slash():
    assert parse_tx_date("2024-07-08").isoformat() == "2024-07-08"
    assert parse_tx_date("08/07/2024").day == 8
    assert parse_tx_date("") is None


def test_summarize_filters_year_and_quarter():
    rows = [
        {"date": "2024-01-10", "description": "Salary", "credit": 1000, "debit": None},
        {"date": "2024-04-02", "description": "Rent", "credit": None, "debit": 400},
        {"date": "2023-04-02", "description": "Old", "credit": None, "debit": 50},
    ]
    q2 = summarize_transactions(rows, year=2024, quarter=2)
    assert q2["count"] == 1
    assert q2["debits"] == 400
    year = summarize_transactions(rows, year=2024, quarter=None)
    assert year["count"] == 2
    assert year["credits"] == 1000
    md = summary_markdown(year)
    assert "2024" in md
    assert "Salary" in md
