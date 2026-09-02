import scipy.sparse.linalg as spla

from beamfem import (
    Material, Model, SciPyLUSolver, Section, UX, available_sparse_solvers,
    factorize_static, get_sparse_solver, register_sparse_solver,
)


class CountingSolver:
    name = "test_counting_splu"

    def __init__(self):
        self.calls = 0

    def factorize(self, matrix):
        self.calls += 1
        return spla.splu(matrix)


def _bar():
    model = Model()
    n0, n1 = model.add_node(0, 0), model.add_node(1, 0)
    model.add_truss(
        n0, n1, Material(200e9, 0.3, 7850),
        Section(A=1e-3, Iy=1e-8, Iz=1e-8, J=1e-8),
    )
    model.fix(n0)
    model.fix(n1, [1, 2, 3, 4, 5])
    model.add_load(n1, UX, 1000)
    return model


def test_public_sparse_solver_object_and_registry_paths():
    solver = CountingSolver()
    result = factorize_static(_bar(), sparse_solver=solver)
    assert result.solver_name == solver.name
    assert solver.calls == 1
    assert result.solve_model(_bar()).node_disp(1)[UX] > 0

    register_sparse_solver(solver.name, solver, replace=True)
    assert solver.name in available_sparse_solvers()
    assert get_sparse_solver(solver.name) is solver
    assert isinstance(get_sparse_solver(), SciPyLUSolver)
