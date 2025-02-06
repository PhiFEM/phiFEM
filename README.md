# $\varphi$-FEM with FEniCSx

A phi-FEM implementation for Poisson-Dirichlet problems with residual a posteriori error estimation.

## Prerequisites

- docker/podman

The docker image is based on FEniCSx.
It contains also some python libraries dependencies.

## Build the image:
Replace `YOUR_ENGINE_HERE` by `docker`, `podman` or your favorite container engine (with similar commands as Docker).
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
cd demo/
```
The `main.py` script have the following parameters:
```bash
python3 main.py TEST_CASE SOLVER
```
where
- `TEST_CASE` is one of the available test cases: `pdt_sines_pyramid`, `lshaped` or `flower`.
- `SOLVER` defines the FE solver (`str` among `FEM` or `phiFEM`),
For each test case, the parameters of the $\varphi$-FEM and of the adaptation loop are in `demo/TEST_CASE/parameters.yaml`.
See a detailed explanation of each parameter below.
Example:
```bash
cd demo/
python3 main.py pdt_sines_pyramid phiFEM 
```

## Launch unit tests (inside the container from the root directory):
```bash
pytest
```

## Details on `parameters.yaml`

- `initial_mesh_size`: the maximum size of the edges of the initial mesh.
- `iterations_number`: the number of iterations of the adaptive refinement loop.
- `refinement_method`: the only choices are `uniform`, for uniform refinement or `H10`, for adaptive refinement steered by the $H^1_0$ semi-norm error estimator.
- `bbox`: the bounding box used to compute the initial background mesh in the $\varphi$-FEM as a $2\times 2$ array, first row: xmin, xmax, second row: ymin, ymax.
- `exact_error`: if `true` computes a higher order approximation of the exact error. **WARNING: to use it with $\varphi$-FEM you must have run a FEM loop before.**
- `marking_parameter`: the parameter of the Dörfler marking strategy.
- `use_fine_space`: if `true`, use the proper space for $u_h = w_h \varphi_h$ i.e. if $w_h$ is of degree $k$ and $\varphi_h$ of degree $l$, then $u_h$ is computed in a finite element space of degree $k+l$. If `false`, $u_h$ is computed in the same finite element space as $\varphi_h$.
- `box_mode`: if `true` unused cells in the $\varphi$-FEM solver are kept during the refinement loop. If `false` they are successively removed at each refinement step.
- `finite_element_degree`: the degree of the finite element space for $w_h$.
- `levelset_degree`: the degree of the finite element space for $\varphi_h$.
- `boundary_detection_degree`: the degree of the finite element space for $\hat \varphi_h$, the level set used to detect the cells cut by $\Gamma_h$ (increasing this value increases the precision of the boundary of $\Omega$ detection but also increases the computational cost).
- `quadrature_degree`: the degree of the quadratures performed during the $\varphi$-FEM solves.
- `stabilization_parameter`: the stabilization coefficient of $\varphi$-FEM, $1$ is a good option.
- `save_output`: if `true` saves the output of the computations.