import io


def test_valid_upload_returns_prediction(client, predictor, make_image):
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(make_image()), "sample.png")},
    )

    assert response.status_code == 200
    assert b"NORMAL" in response.data
    assert b"0.25" in response.data
    assert predictor.calls == 1


def test_missing_file_returns_400(client, predictor):
    response = client.post("/predict", data={})

    assert response.status_code == 400
    assert b"Please select an image file." in response.data
    assert predictor.calls == 0


def test_empty_filename_returns_400(client, predictor):
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(b""), "")},
    )

    assert response.status_code == 400
    assert predictor.calls == 0


def test_invalid_image_returns_400(client, predictor):
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(b"not an image"), "fake.png")},
    )

    assert response.status_code == 400
    assert b"not a readable image" in response.data
    assert predictor.calls == 0


def test_unsupported_format_returns_400(client, predictor, make_image):
    response = client.post(
        "/predict",
        data={
            "file": (
                io.BytesIO(make_image(image_format="GIF")),
                "renamed.png",
            )
        },
    )

    assert response.status_code == 400
    assert b"PNG or JPEG" in response.data
    assert predictor.calls == 0


def test_excessive_dimensions_return_400(client, predictor, make_image):
    response = client.post(
        "/predict",
        data={
            "file": (
                io.BytesIO(make_image(size=(101, 100))),
                "large.png",
            )
        },
    )

    assert response.status_code == 400
    assert b"dimensions are too large" in response.data
    assert predictor.calls == 0


def test_oversized_request_returns_413(app, client, predictor):
    app.config["MAX_CONTENT_LENGTH"] = 1024

    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(b"x" * 2048), "large.png")},
    )

    assert response.status_code == 413
    assert b"request size limit" in response.data
    assert predictor.calls == 0
