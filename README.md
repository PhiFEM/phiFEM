# phiFEM: a convenience package for using φ-FEM with FEniCSx

φ-FEM (or phiFEM) is an immersed boundary finite element method leveraging levelset functions to avoid the use of any non-standard finite element spaces or non-standard quadrature rules.
More information about φ-FEM can be found in the various publications (see e.g. [^1] and [^2]).

This package provides convenience tools that helps with the implementation of φ-FEM schemes in the [FEniCSx](https://fenicsproject.org/) computation platform.

[^1]: M. DUPREZ and A. LOZINSKI, $\phi$-FEM: A finite element method on domains defined by level-sets, SIAM J. Numer. Anal., 58 (2020), pp. 1008-1028, [https://epubs.siam.org/doi/10.1137/19m1248947](https://epubs.siam.org/doi/10.1137/19m1248947)
[^2]: S. COTIN, M. DUPREZ, V. LLERAS, A. LOZINSKI, and K. VUILLEMOT, $\phi$-FEM: An efficient simulation tool using simple meshes for problems in structure mechanics and heat transfer, Partition of Unity Methods, (2023), pp. 191-216, [https://www.semanticscholar.org/paper/%CF%86-FEM%3A-an-efficient-simulation-tool-using-simple-in-Cotin-Duprez/82f2015ac98f66af115ae57f020b0b1a45c46ad0](https://www.semanticscholar.org/paper/%CF%86-FEM%3A-an-efficient-simulation-tool-using-simple-in-Cotin-Duprez/82f2015ac98f66af115ae57f020b0b1a45c46ad0),

## Prerequisites

### General

- [dolfinx](https://github.com/FEniCS/dolfinx)

  
### To run some of the demos

- [PyYAML](https://pypi.org/project/PyYAML/)

### To run the tests

- [pytest](https://docs.pytest.org/en/stable/)

## Usage

We recommend to use `phiFEM` inside the `dolfinx` container (e.g. `ghcr.io/fenics/dolfinx/dolfinx:stable`).

- Launch the `dolfinx` container in interactive mode using, e.g. [Docker](https://www.docker.com/) (see [the docker documentation](https://docs.docker.com/reference/cli/docker/container/run/) for the meaning of the different arguments):
  `docker run -ti -v $(pwd):/home/dolfinx/shared -w /home/dolfinx/shared dolfinx/dolfinx:stable`
- Inside the container install the phiFEM package with `pip`:
  `pip install phifem` 

## License

`phiFEM` is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with `phiFEM`. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

## Authors (alphabetical)

Raphaël Bulle ([https://rbulle.github.io](https://rbulle.github.io/))  
Michel Duprez ([https://michelduprez.fr/](https://michelduprez.fr/))  
Killian Vuillemot
