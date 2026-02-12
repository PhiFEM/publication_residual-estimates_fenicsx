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

1. **lshaped (Test case I):** Homogeneous zero Dirichlet boundary condition on a L-shaped domain.
2. **circle (Test case II):** Non-homogeneous Dirichlet boundary condition on a circular domain. The solution as the following analytical expression: $u(x,y) = \big((x-1)^2 + y^2\big)^{1/3}$.
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

### Generate the results from all pre-defined schemes for a specific test case

The test case parameters are defined in the `demo/TC/data.py` file.

To generate the results for a specific test case, inside the container run:
   
   ```bash
   cd demo/
   bash launch-test-case.sh TC
   ```
with `TC=lshaped,circle,drop` or `drop_bc`.  
The results are located in `demo/TC/output_*/`.

### Run an AFEM loop for a specific scheme and a specific test case

Schemes parameters are specified in `.yaml` files in each demo directory, e.g. the file `demo/lshaped/phifem-bc-geo.yaml` contains the parameters for the phiFEM adaptive refinement scheme steered by $\eta$.  
It is possible to run a specific scheme via:

```bash
cd demo/
python main.py TC/SCHEME
```
where `TC=lshaped,circle,drop` or `drop_bc` and `SCHEME` is the name of a `.yaml` file.  

> For example:
> ```bash
> cd demo/
> python main.py circle/phifem-geo
> ```
> runs an adaptive refinement loop on the circle test case with the phiFEM scheme and $\eta_{\overline{BC}}$ estimator.

### Custom test case and custom scheme

It is possible to create a custom test case and a custom scheme by creating the proper file hierarchy.  

#### Custom test case

First create the directory for the custom test case:
```bash
cd demo/
mkdir my-custom-tc/
```
then, create a `data.py` file with the following mandatory parameters:
```python
INITIAL_MESH_SIZE = # your custom initial mesh size.
MAXIMUM_DOF = # your custom maximum number of dof.
REFERENCE = # this is the name of the scheme used to generate de reference solution.
MAX_EXTRA_STEP_ADAP = # your custom number of additional adaptive refinement steps in order to compute a reference solution.
MAX_EXTRA_STEP_ADAP = # your custom number of additional uniform refinement steps in order to compute a reference solution.

def generate_levelset(mode):
   def levelset(x):
      # your custom levelset function (mode is standing either for numpy or ufl, each one might be used at some point in the scripts).
      return value
   return levelset

def gen_mesh(hmax, curved=False):
   import ngsPETSc.utils.fenicsx as ngfx
   from mpi4py import MPI
   from netgen.geom2d import SplineGeometry

   # your custom geometry and mesh defined using NetGen API.
   return mesh, geoModel

def generate_dirichlet_data(mode):
   def dirichlet_data(x):
      # your custom Dirichlet BC data
      return value
   return dirichlet_data
```

#### Custom scheme

To define a custom scheme, you must add a `.yaml` file to a test case directory.
For example, in the custom test case directory we created above, create the file `demo/my-custom-tc/my-custom-scheme.yaml` containing:
```yaml
refinement: # your custom refinement scheme "adap" or "unif"
mesh_type: # your custom phiFEM mesh type "bg" to keep the background mesh, "sub" to compute the submesh at each step.
finite_element_degree: # your custom FE degree for the solution u
auxiliary_degree: # your custom FE degree for the auxiliary phiFEM unknown p
levelset_degree: # your custom FE degree for the levelset Lagrange interpolation
boundary_detection_degree: # your custom quadrature degree for the phiFEM boundary detection and mesh tagging
boundary_correction: # your custom boundary correction True if you want to use \eta_geo in the estimator, False otherwise.
dirichlet_estimator: # your custom Dirichlet estimator True if you want to use \eta_BC in the estimator, False otherwise.
penalization_coefficient: # your custom phiFEM penalization coefficient.
stabilization_coefficient: # your custom phiFEM stabilization coefficient.
marking_parameter: # your custom Dörfler's marking parameter.
discretize_levelset: # your custom discretization of the levelset during the marking procedure of phiFEM True if you want phiFEM to mark the cells from the levelset interpolated in a FE space, False if you want phiFEM to mark the cells from the levelset discretized as a UFL function.
single_layer: # your custom phiFEM cut cells layer True if you want to restrict phiFEM to a single layer of cut cells, False otherwise.
bbox: # your custom bounding box to define the background mesh.
name: # your custom scheme name.
```

Once your custom test case and custom scheme are defined you should be able to run:
```bash
cd demo/
python main.py my-custom-tc/my-custom-scheme
```
and get the results in `demo/my-custom-tc/output_my-custom-scheme`.

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
