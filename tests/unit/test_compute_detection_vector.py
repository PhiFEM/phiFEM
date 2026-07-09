import dolfinx as dfx
from dolfinx.io import XDMFFile
import numpy as np
import pytest
from basix.ufl import element
from mpi4py import MPI
import basix

from phifem.tags import _compute_detection_vector
from phifem.measures import detection

def _interior_test(x):
    return np.isclose(x, -1.)

def _exterior_test(x):
    return np.isclose(x, 1.)

def _cut_test(x):
    return np.logical_and(x > -1., x < 1.)

mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
all_cells = dfx.mesh.locate_entities(mesh, mesh.topology.dim, lambda x: np.ones_like(x[0]).astype(bool))
bdy_facets = dfx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.ones_like(x[0]).astype(bool))

test_mesh = [mesh]

def levelset_1(x):
    return (x[0]- 0.5)**2+x[1]**2-0.2**2

benchmark_cells_1_1 = np.array([1.] * len(all_cells))
benchmark_cells_1_2 = np.array([0.5, 0.5, 0.5, 1., 1., 1., 1., 1.])
benchmark_facets_1_1 = np.array([1.] * len(bdy_facets))

# data = (levelset, measure_degree, cell_type, measure_integral_type, benchmark)
data_1 = (levelset_1, 0, "triangle", "dx", benchmark_cells_1_1)
data_2 = (levelset_1, 1, "triangle", "dx", benchmark_cells_1_2)
data_3 = (levelset_1, 2, "triangle", "dx", benchmark_cells_1_2)
# data_4 = (levelset_1, 0, "segment",  "ds", benchmark_1)
# data_5 = (levelset_1, 1, "segment",  "ds", benchmark_1)
# data_6 = (levelset_1, 2, "segment",  "ds", benchmark_1)
test_data = [data_1, data_2, data_3] #, data_4, data_5, data_6]

@pytest.mark.parametrize("mesh", test_mesh)
@pytest.mark.parametrize("levelset, measure_degree, cell_type, measure_integral_type, benchmark", test_data)
def test_compute_detection_vector(mesh, levelset, cell_type, measure_degree, measure_integral_type, benchmark):
    detection_measure = detection(mesh, measure_degree, cell_type, measure_integral_type)

    cell_name = mesh.topology.cell_name()
    levelset_element = element("Lagrange", cell_name, 1)
    levelset_space = dfx.fem.functionspace(mesh, levelset_element)
    discrete_levelset = dfx.fem.Function(levelset_space)
    discrete_levelset.interpolate(levelset)

    detection_vector = _compute_detection_vector(mesh, discrete_levelset, detection_measure)
    if cell_type == "triangle":
        all_entities = all_cells
    elif cell_type == "segment":
        all_entities = bdy_facets

    assert len(detection_vector) == len(all_entities)

    for detection_test in [_interior_test, _exterior_test, _cut_test]:
        assert np.all(detection_test(detection_vector) == detection_test(benchmark))

if __name__=="__main__":
    test_compute_detection_vector(test_mesh + data_4)