# Phi-FEM with FEniCSx

A phi-FEM implementation for Poisson-Dirichlet problems.

## Prerequisites

- docker/podman

The docker image is based on FEniCSx.
It contains also some python libraries dependencies.

## Build the image:
Replace `YOUR_ENGINE_HERE` by `docker`, `podman` or your favorite container engine.
```bash
export CONTAINER_ENGINE="YOUR_ENGINE_HERE"
${CONTAINER_ENGINE} login
cd docker/
sudo -E bash build_image.sh
```

## Launch the image (from the root directory):
```bash
sudo -E bash run_image.sh
```

## Run an example (inside the container from the root directory):
```bash
cd phiFEM/homogeneous_dirichlet/
python3 main.py
```

## Launch unit tests (inside the container from the root directory):
```bash
pytest
```

[Test](phiFEM/homogeneous_dirichlet/profile_bg_mesh_static.html)