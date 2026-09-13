import logging
import re

import pytest


@pytest.mark.parametrize(
    "method, path, expected_status",
    [
        ("GET", "/health/live", 200),
        ("POST", "/api/v1/predict", 400),
    ],
)
def test_request_log_matches_response(
    app,
    client,
    caplog,
    method,
    path,
    expected_status,
):
    caplog.clear()

    with caplog.at_level(logging.INFO, logger=app.logger.name):
        response = client.open(path, method=method)

    assert response.status_code == expected_status

    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == app.logger.name
        and record.getMessage().startswith("request_completed ")
    ]

    assert len(messages) == 1

    fields = dict(
        field.split("=", 1)
        for field in messages[0].split()[1:]
    )

    assert fields["request_id"] == request_id
    assert fields["method"] == method
    assert fields["path"] == path
    assert fields["status"] == str(expected_status)
    assert re.fullmatch(r"\d+\.\d{2}", fields["duration_ms"])
    assert float(fields["duration_ms"]) >= 0


def test_each_request_gets_a_different_id(client):
    first_response = client.get("/health/live")
    second_response = client.get("/health/live")

    first_id = first_response.headers["X-Request-ID"]
    second_id = second_response.headers["X-Request-ID"]

    assert re.fullmatch(r"[0-9a-f]{32}", first_id)
    assert re.fullmatch(r"[0-9a-f]{32}", second_id)
    assert first_id != second_id
