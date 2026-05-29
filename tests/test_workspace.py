"""出力先 workspace フォルダの検証。"""

import os

from beamfem import Material, Section, Model, solve_static, recover_forces, UY
from beamfem import workspace

STEEL = Material(E=200e9, nu=0.3)


def _forces():
    m = Model()
    a = m.add_node(0, 0, 0)
    b = m.add_node(2, 0, 0)
    m.add_element(a, b, STEEL, Section.rectangle(b=0.05, h=0.1))
    m.fix(a)
    m.add_load(b, UY, -1000)
    return recover_forces(m, solve_static(m))


def test_relative_path_goes_to_workspace(tmp_path):
    workspace.set_workspace(str(tmp_path / "ws"))
    try:
        path = _forces().to_csv("out.csv", items=["Mz"])
        assert path == os.path.join(str(tmp_path / "ws"), "out.csv")
        assert os.path.exists(path)
    finally:
        workspace.set_workspace(workspace._DEFAULT)


def test_absolute_path_respected(tmp_path):
    target = str(tmp_path / "abs.csv")
    path = _forces().to_csv(target, items=["Mz"])
    assert path == target
    assert os.path.exists(path)


def test_default_workspace_name():
    workspace.set_workspace(workspace._DEFAULT)
    assert workspace.get_workspace() == "workspace"
    # 解決パスは workspace/ 配下
    assert workspace.resolve("a.csv", create=False) == os.path.join("workspace", "a.csv")
