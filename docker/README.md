# Run with Docker

Place the trained model at `model/cnn_best_100.h5`. From the repository root,
build the image:

```bash
docker build -f docker/Dockerfile -t pneumonia-detector .
```

Run the container with the model directory mounted read-only:

```bash
docker run --rm --name pneumonia-detector -p 127.0.0.1:8000:8000 \
  --mount type=bind,source="$(pwd)/model",target=/app/model,readonly \
  pneumonia-detector
```

Open http://localhost:8000 once the model has loaded. Stop the container from
another terminal with `docker stop pneumonia-detector`.

The image uses Python 3.12 and Gunicorn with one worker, runs as a non-root
user, and performs inference on the CPU. The model is not included in the image;
its file and parent directory must be readable by the container user. The
container health check calls `/health/ready`.

To use another host port, change `127.0.0.1:8000:8000` to
`127.0.0.1:8080:8000`, then open http://localhost:8080.
