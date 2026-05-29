"""Ground Structure 法（トラスLP）トポロジー最適化の検証。"""

import numpy as np

from beamfem.optimize import (
    GroundStructure,
    solve_min_volume,
    generate_members,
    grid_nodes,
    equilibrium_matrix,
)

SIGMA = 200e6
P = 10000.0


def test_single_bar_volume():
    """支点→自由節点の単一バー: 体積 = P L / σ、軸力 = P（引張）。"""
    gs = GroundStructure(
        nodes=np.array([[0.0, 0.0], [2.0, 0.0]]),
        members=[(0, 1)],
        supports={0: [0, 1]},
        load_cases=[{(1, 0): P}],
    )
    res = solve_min_volume(gs, SIGMA)
    assert np.isclose(res.volume, P * 2.0 / SIGMA, rtol=1e-6)
    assert np.isclose(res.forces[0, 0], P, rtol=1e-6)


def test_symmetric_two_bar():
    """対称2バー静定: 体積 = 2P/σ、各軸力 = P/√2。"""
    gs = GroundStructure(
        nodes=np.array([[0.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]]),
        members=[(0, 1), (0, 2)],
        supports={1: [0, 1], 2: [0, 1]},
        load_cases=[{(0, 0): P}],
    )
    res = solve_min_volume(gs, SIGMA)
    assert np.isclose(res.volume, 2 * P / SIGMA, rtol=1e-6)
    assert np.allclose(res.forces[0], P / np.sqrt(2), rtol=1e-6)


def test_redundant_picks_direct_bar():
    """冗長な候補から、荷重と一直線の最短バーのみを選ぶ（大域最適）。"""
    gs = GroundStructure(
        nodes=np.array([[0.0, 0.0], [-1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]]),
        members=[(0, 1), (0, 2), (0, 3), (2, 3)],
        supports={1: [0, 1], 2: [0, 1], 3: [0, 1]},
        load_cases=[{(0, 0): P}],
    )
    res = solve_min_volume(gs, SIGMA)
    assert np.isclose(res.volume, P * 1.0 / SIGMA, rtol=1e-6)
    assert list(res.active()) == [0]


def test_equilibrium_satisfied():
    """解の部材軸力が節点平衡 B n = f を満たす。"""
    nodes = grid_nodes(4, 3, 3.0, 2.0)
    members = generate_members(nodes)
    supports = {iy * 4 + 0: [0, 1] for iy in range(3)}
    load_node = 1 * 4 + 3
    gs = GroundStructure(nodes, members, supports, [{(load_node, 1): -P}])
    res = solve_min_volume(gs, SIGMA)
    from beamfem.optimize.topology import _load_vector

    B, free, dof_index = equilibrium_matrix(gs)
    f = _load_vector(gs, gs.load_cases[0], dof_index, len(free))
    assert np.allclose(B @ res.forces[0], f, atol=1e-6)


def test_multiple_load_cases():
    """断面は最悪荷重ケースで決まる。"""
    nodes = np.array([[0.0, 0.0], [2.0, 0.0]])
    gs = GroundStructure(
        nodes, [(0, 1)], supports={0: [0, 1]},
        load_cases=[{(1, 0): P}, {(1, 0): -2 * P}],  # 引張P と 圧縮2P
    )
    res = solve_min_volume(gs, SIGMA)
    # A は |2P|/σ で決まる -> 体積 = 2P L/σ
    assert np.isclose(res.volume, 2 * P * 2.0 / SIGMA, rtol=1e-6)


def test_generate_members_removes_collinear():
    """一直線上の3節点で、両端を結ぶ長い部材は除外される。"""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    members = set(generate_members(nodes))
    assert (0, 1) in members and (1, 2) in members
    assert (0, 2) not in members  # 節点1を貫くため冗長


def test_different_tension_compression():
    """引張/圧縮で許容応力が異なる場合の体積。"""
    nodes = np.array([[0.0, 0.0], [2.0, 0.0]])
    # 圧縮許容を半分にすると、圧縮材の所要断面は倍 -> 体積も倍
    gs = GroundStructure(nodes, [(0, 1)], supports={0: [0, 1]}, load_cases=[{(1, 0): -P}])
    res = solve_min_volume(gs, sigma_t=SIGMA, sigma_c=SIGMA / 2)
    assert np.isclose(res.volume, P * 2.0 / (SIGMA / 2), rtol=1e-6)
