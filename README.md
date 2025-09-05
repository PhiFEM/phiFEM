# $\varphi$-FEM with FEniCSx

$\varphi$-FEM is an immersed boundary finite element method leveraging levelset functions to avoid the use of any non-standard finite element spaces or non-standard quadrature rules.
More information about $\varphi$-FEM can be found in the various publications (see e.g. [^1] and [^2]).

This repository aims at providing an implementation of the $\varphi$-FEM in the [FEniCSx](https://fenicsproject.org/) computation platform as well as several demos to illustrate its capabilities.

[^1]: M. DUPREZ and A. LOZINSKI, $\phi$-FEM: A finite element method on domains defined by level-sets, SIAM J. Numer. Anal., 58 (2020), pp. 1008-1028, [https://epubs.siam.org/doi/10.1137/19m1248947](https://epubs.siam.org/doi/10.1137/19m1248947)
[^2]: S. COTIN, M. DUPREZ, V. LLERAS, A. LOZINSKI, and K. VUILLEMOT, $\phi$-FEM: An efficient simulation tool using simple meshes for problems in structure mechanics and heat transfer, Partition of Unity Methods, (2023), pp. 191-216, [https://www.semanticscholar.org/paper/%CF%86-FEM%3A-an-efficient-simulation-tool-using-simple-in-Cotin-Duprez/82f2015ac98f66af115ae57f020b0b1a45c46ad0](https://www.semanticscholar.org/paper/%CF%86-FEM%3A-an-efficient-simulation-tool-using-simple-in-Cotin-Duprez/82f2015ac98f66af115ae57f020b0b1a45c46ad0),

## Prerequisites

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)/[podman](https://podman.io/)

The docker image is based on the stable dolfinx image (see [FEniCSx](https://fenicsproject.org/)).

## Usage

### Build the image (from the root directory):
Replace `YOUR_ENGINE_HERE` by `docker`, `podman` or your favorite container engine (the following instructions use Docker/podman UI).
```bash
export CONTAINER_ENGINE="YOUR_ENGINE_HERE"
cd docker/
bash build_image.sh
```

### Launch the image (from the root directory):
```bash
bash run_image.sh
```

### Run an example (inside the container from the root directory):
```bash
cd demo/weak-dirichlet/flower
```
Run the `main.py` script:
```bash
python main.py bg
```

## License

`PhiFEM/Poisson-Dirichlet-fenicsx` is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with `PhiFEM/Poisson-Dirichlet-fenicsx`. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

## Authors (alphabetical)

Raphaël Bulle ([https://rbulle.io](https://rbulle.github.io/))  
Michel Duprez ([https://michelduprez.fr/](https://michelduprez.fr/))  
Killian Vuillemot  