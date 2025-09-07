# phiFEM: a convenience package for using φ-FEM with FEniCSx

This package provides convenience tools that help with the implementation of φ-FEM schemes in the [FEniCSx](https://fenicsproject.org/) computation platform.

φ-FEM (or phiFEM) is an immersed boundary finite element method leveraging levelset functions to avoid the use of any non-standard finite element spaces or non-standard quadrature rules near the boundary of the domain.
More information about φ-FEM can be found in the various publications (see e.g. [^1] and [^2]).

[^1]: M. DUPREZ and A. LOZINSKI, $\phi$-FEM: A finite element method on domains defined by level-sets, SIAM J. Numer. Anal., 58 (2020), pp. 1008-1028, [https://epubs.siam.org/doi/10.1137/19m1248947](https://epubs.siam.org/doi/10.1137/19m1248947)  
[^2]: S. COTIN, M. DUPREZ, V. LLERAS, A. LOZINSKI, and K. VUILLEMOT, $\phi$-FEM: An efficient simulation tool using simple meshes for problems in structure mechanics and heat transfer, Partition of Unity Methods, (2023), pp. 191-216, [https://www.semanticscholar.org/paper/%CF%86-FEM%3A-an-efficient-simulation-tool-using-simple-in-Cotin-Duprez/82f2015ac98f66af115ae57f020b0b1a45c46ad0](https://www.semanticscholar.org/paper/%CF%86-FEM%3A-an-efficient-simulation-tool-using-simple-in-Cotin-Duprez/82f2015ac98f66af115ae57f020b0b1a45c46ad0),

## Prerequisites

- [dolfinx](https://github.com/FEniCS/dolfinx) >= 0.9.0

## Usage

We recommend to use `phiFEM` inside the `dolfinx` container (e.g. `ghcr.io/fenics/dolfinx/dolfinx:stable`).

- Launch the `dolfinx` container in interactive mode using, e.g. [Docker](https://www.docker.com/) (see [the docker documentation](https://docs.docker.com/reference/cli/docker/container/run/) for the meaning of the different arguments):  
  ```bash
  docker run -ti -v $(pwd):/home/dolfinx/shared -w /home/dolfinx/shared dolfinx/dolfinx:stable
  ```
- Inside the container install the phiFEM package with `pip`:  
  ```bash
  pip install phifem
  ``` 

## Run the demos

The demos can be found on the [phiFEM Github repository](https://github.com/PhiFEM/phiFEM).
To run the demos you'll need to clone the repository and build and launch the container:

1) Clone the phiFEM repository:
  
  ```bash
  git clone https://github.com/PhiFEM/phiFEM.git
  ```

2) Build the container (you might need sudo privileges):

  ```bash
  cd phifem/docker
  bash build_image.sh
  ```

3) Run the container (you might need sudo privileges):

  ```bash
  cd ../
  bash run_image.sh
  ```

4) Inside the container navigate the demo directory and e.g. run the interface elasticity demo:

  ```bash
  cd demo/interface-elasticity
  python main param1
  ```

## Run the tests

To run the tests, follow the steps 1 to 3 to run the demos and inside the container:

```bash
cd tests
pytest
```

## License

`phiFEM` is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with `phiFEM`. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

## Authors (alphabetical)

Raphaël Bulle ([https://rbulle.github.io](https://rbulle.github.io/))  
Michel Duprez ([https://michelduprez.fr/](https://michelduprez.fr/))  
Killian Vuillemot
