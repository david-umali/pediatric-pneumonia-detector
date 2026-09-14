# Run with Docker Compose

Install Docker with the Compose plugin and place the trained model at
`model/cnn_best_100.h5`. Run the following commands from the repository root.

Build the image and start the app in the background:

```bash
docker compose -f docker/compose.yaml up --build -d
```

Open http://localhost:8000 once the model has loaded. Check container status,
follow logs, or check readiness:

```bash
docker compose -f docker/compose.yaml ps
docker compose -f docker/compose.yaml logs -f app
curl -i http://localhost:8000/health/ready
```

Stop and remove the container and Compose network:

```bash
docker compose -f docker/compose.yaml down
```

The Compose service `app` builds `pneumonia-detector:latest` using the repository
root as its build context and `docker/Dockerfile` as its Dockerfile. It binds
port 8000 to the host's loopback address, making the app accessible locally.

The existing `model/` directory is mounted read-only at `/app/model`. Compose
will not create a missing source directory. The configuration sets `MODEL_PATH`
to `/app/model/cnn_best_100.h5` and `MODEL_VERSION` to `cnn_best_100`; update these
values in `compose.yaml` when using a different model. Restart the app after
replacing model weights so it loads the new file:

```bash
docker compose -f docker/compose.yaml restart app
```

After changing Compose configuration, run `up -d` again to apply it. Use
`up --build -d` after changing application code or image dependencies.

The `unless-stopped` restart policy restarts the container after it exits unless
it has been explicitly stopped. A failed health check alone does not restart it.

To use another host port, change the mapping in `compose.yaml` to
`127.0.0.1:8080:8000`, run `up -d` again, and open http://localhost:8080.

## Run without Compose

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
