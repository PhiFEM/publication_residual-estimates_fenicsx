# Residual-based a posteriori error estimates with boundary correction for $\varphi$-FEM

This repository contains the code used to generate the numerical results in the study:

*Residual-based a posteriori error estimates with boundary correction for φ-FEM*  
R. Becker, R. Bulle, M. Duprez, V. Lleras  
[https://hal.science/hal-04931977](https://hal.science/hal-04931977)  

## This repository is for reproducibility purposes only
It is "frozen in time" and not maintained.
To use our $\varphi$-FEM code please refer to the [phiFEM repository](https://github.com/PhiFEM/Poisson-Dirichlet-fenicsx).

## Test cases

This repository contains 4 test cases 

1. **lshaped (Test case I):** Homogeneous zero Dirichlet boundary condition on a tilted L-shaped domain.
2. **circle (Test case II):** Non-homogeneous Dirichlet boundary condition on a circular domain. The solution as the following analytical expression: $u(x,y) = (x^2 + y^2)^{1/3}$.
3. **drop (Test case III):** Homogeneous zero Dirichlet boundary condition on a "drop"-shaped domain (with a polynomial cusp point).
4. **drop_BC:** Non-homogeneous Dirichlet boundary condition on a "drop"-shaped domain (with a polynomial cusp point). The Dirichlet data is a Gaussian function centered at the cusp point.

## Generate the results

### Prerequisites

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/)/[podman](https://podman.io/)

The image is based on [FEniCSx](https://fenicsproject.org/) with additional python libraries dependencies (see `docker/requirements.txt`).

### Install and launch the residual-phifem container

1) Clone this repository in a dedicated directory:
   
   ```bash
   mkdir residual-phifem/
   git clone https://github.com/PhiFEM/publication_residual-estimates_fenicsx.git residual-phifem
   ```

2) Download the image from the docker.io registry, in the main directory:
   
   ```bash
   export CONTAINER_ENGINE=docker
   cd residual-phifem
   sudo -E bash pull_image.sh
   ```

3) Launch the container:

   ```bash
   sudo -E bash run_image.sh
   ```

### Generate the results from all schemes for a specific test case

Inside the container:
   
   ```bash
   cd demo/
   bash launch-test-case.sh DEMO
   ```
with `DEMO=lshaped,circle,drop` or `drop_bc`.  
The ²results are located in `demo/DEMO/output_*/`.

### Generate the results for a specific scheme for a specific test case

Schemes parameters are specified in `.yaml` files in each demo directory, e.g. the file `demo/lshaped/phifem-bc-geo.yaml` contains the parameters for the phiFEM adaptive refinement scheme steered by $\eta$.  
It is possible to run a specific scheme via:

```bash
cd demo/
python3 main.py DEMO/SCHEME
```
where `DEMO=lshaped,circle,drop` or `drop_bc` and `SCHEME` is the name of a specific parameter.

Other parameters can be modified in the `.yaml` files:

```bash
demo
└── DEMO_NAME
    └── parameters.yaml
```

> **Remark:** `python3 main.py flower FEM` does not work due to the absence of refinement strategy for curved boundaries.

## Issues and support

Please use the issue tracker to report any issues.

## Authors (alphabetical)

[Roland Becker](https://lma-umr5142.univ-pau.fr/fr/organisation/membres/cv_-becker-fr.html), Université de Pau et des Pays de l'Adour  
[Raphaël Bulle](https://rbulle.github.io/), Inria Nancy Grand-Est  
[Michel Duprez](https://michelduprez.fr/), Inria Nancy Grand-Est  
[Vanessa Lleras](https://vanessalleras.wixsite.com/lleras), Université de Montpellier

## License

residual-phifem is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with residual-phifem. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
