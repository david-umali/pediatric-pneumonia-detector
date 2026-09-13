import pytest


@pytest.mark.parametrize(
    "endpoint, expected_status",
    [
        ("/health/live", "alive"),
        ("/health/ready", "ready"),
    ],
)
def test_health_endpoint_returns_expected_response(
    client,
    endpoint,
    expected_status,
):
    response = client.get(endpoint)

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == {"status": expected_status}


def test_health_checks_do_not_run_inference(client, predictor):
    client.get("/health/live")
    client.get("/health/ready")

    assert predictor.calls == 0
