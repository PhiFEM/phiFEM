<p align="center" width="100%">
    <img width="100%" src="https://github.com/PhiFEM/phiFEM/raw/main/logo/Logo_PhiFEM_wider.png">
</p>

# phiFEM: a convenience package for using $\varphi$-FEM with FEniCSx

This package provides convenience tools that help with the implementation of $\varphi$-FEM schemes in the [FEniCSx](https://fenicsproject.org/) computation platform.

$\varphi$-FEM (or phiFEM) is an immersed boundary finite element method leveraging levelset functions to avoid the use of any non-standard finite element spaces or non-standard quadrature rules near the boundary of the domain.
More information about $\varphi$-FEM can be found in the various publications (see e.g. [^1] and [^2]).

[^1]: M. DUPREZ and A. LOZINSKI, $\varphi$-FEM: A finite element method on domains defined by level-sets, SIAM J. Numer. Anal., 58 (2020), pp. 1008-1028, [https://epubs.siam.org/doi/10.1137/19m1248947](https://epubs.siam.org/doi/10.1137/19m1248947)  
[^2]: S. COTIN, M. DUPREZ, V. LLERAS, A. LOZINSKI, and K. VUILLEMOT, $\varphi$-FEM: An efficient simulation tool using simple meshes for problems in structure mechanics and heat transfer, Partition of Unity Methods, (2023), pp. 191-216, [https://www.semanticscholar.org/paper/%CF%86-FEM%3A-an-efficient-simulation-tool-using-simple-in-Cotin-Duprez/82f2015ac98f66af115ae57f020b0b1a45c46ad0](https://www.semanticscholar.org/paper/%CF%86-FEM%3A-an-efficient-simulation-tool-using-simple-in-Cotin-Duprez/82f2015ac98f66af115ae57f020b0b1a45c46ad0).

## [phiFEM Pypi webpage](https://pypi.org/project/phiFEM/)

## Dependencies

See [phifem-env.yml](phifem-env.yml).

> ⚠️ Note that all these dependencies are already included in `dolfinx` Docker container.

## Installation

### Conda environment

- Create a `phifem` Conda environment from the spec file `phifem-env.yml`:
  ```bash
  conda create -f phifem-env.yml
  ```
- Activate the `phifem` environment:
  ```bash
  conda activate phifem
  ```

### Docker container

The `phiFEM` package can be used inside the `dolfinx` container (e.g. `ghcr.io/fenics/dolfinx/dolfinx:stable`)

- Launch the `dolfinx` container in interactive mode using, e.g. [Docker](https://www.docker.com/) (see [the docker documentation](https://docs.docker.com/reference/cli/docker/container/run/) for the meaning of the different arguments):  
  ```bash
  docker run -ti -v $(pwd):/home/dolfinx/shared -w /home/dolfinx/shared dolfinx/dolfinx:stable
  ```
- Inside the container install `phiFEM` via pip:  
  ```bash
  pip install phifem
  ``` 

## Run the demos

The demos can be found on the [phiFEM Github repository](https://github.com/PhiFEM/phiFEM/tree/main/demo).
To run the demos first clone the repository.

**Inside the Docker container/Conda environment:**

- Install the demos dependencies:

  ```bash
  pip install polars Pyaml
  ```

- Navigate the demo directory and run it e.g.:

  ```bash
  cd demo/weak-dirichlet/flower
  python main.py bg
  ```

> ⚠️ The demo files require arguments, for more info run `python main.py -h`.

## Run the tests

**Inside the Docker container/Conda environment:**

- Install `pytest`:  
  ```bash
  pip install pytest
  ```

- Run the tests:  
  ```bash
  cd tests
  pytest
  ```

## License

`phiFEM` is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with `phiFEM`. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

## Logo credit

Frédérique Lecourtier ([https://mimesis.inria.fr/members/frederique-lecourtier/](https://mimesis.inria.fr/members/frederique-lecourtier/))