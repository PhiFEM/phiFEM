#!/bin/bash
${CONTAINER_ENGINE} run --privileged -ti -v $(pwd):/home/dolfinx/shared -w /home/dolfinx/shared rbulle/residual-phifem:latest