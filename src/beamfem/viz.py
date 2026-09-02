"""解析結果の図示（matplotlib）。

- ``plot_model``    : 変形前のモデル（節点・要素・支持・荷重）
- ``plot_deformed`` : 変形図（要素ごとに形状関数で滑らかに補間）

2D（全節点 z=0 かつ面外変位が無視できる）か 3D かを自動判定する。
matplotlib は任意依存（``pip install -e ".[viz]"``）。
"""

from __future__ import annotations

import numpy as np

from .assembly import element_dof_map
from .element3d import rotation_matrix, transformation_matrix
from .model import DOF_PER_NODE, UX, UY, UZ, RX, RY, RZ, Model, TrussElement
from .solver import StaticResult


def _draw_shell_faces(ax, tris, planar, **kw):
    """三角形頂点リスト tris (各 (3,3) 配列) を面として描く。"""
    if not tris:
        return
    if planar:
        from matplotlib.collections import PolyCollection

        polys = [t[:, :2] for t in tris]
        ax.add_collection(PolyCollection(polys, **kw))
    else:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        ax.add_collection3d(Poly3DCollection(tris, **kw))


# ----------------------------------------------------------------------
# 変形形状の補間
# ----------------------------------------------------------------------
def _hermite(xi: np.ndarray, L: float):
    """エルミート3次形状関数 (xi in [0,1])。

    たわみ v(xi) = H1 v1 + L*H2 t1 + H3 v2 + L*H4 t2
    （t は端部回転）。変形図の描画用で曲げの曲率を滑らかに表現する。
    """
    H1 = 1 - 3 * xi**2 + 2 * xi**3
    H2 = xi - 2 * xi**2 + xi**3
    H3 = 3 * xi**2 - 2 * xi**3
    H4 = -xi**2 + xi**3
    return H1, L * H2, H3, L * H4


def element_deformed_curve(
    p1: np.ndarray,
    p2: np.ndarray,
    u_elem_global: np.ndarray,
    vref=None,
    scale: float = 1.0,
    n: int = 12,
) -> np.ndarray:
    """1要素の変形後曲線を全体座標 (n, 3) で返す。

    u_elem_global は当該要素の 12 自由度変位（全体座標）。
    """
    L = float(np.linalg.norm(p2 - p1))
    R = rotation_matrix(p1, p2, vref)
    T = transformation_matrix(R)
    d_local = T @ u_elem_global  # 局所座標の節点変位

    u1x, u1y, u1z, _, ry1, rz1 = d_local[0:6]
    u2x, u2y, u2z, _, ry2, rz2 = d_local[6:12]

    xi = np.linspace(0.0, 1.0, n)
    H1, H2, H3, H4 = _hermite(xi, L)

    # 軸方向は線形
    ux = (1 - xi) * u1x + xi * u2x
    # 局所y方向のたわみ（回転は theta_z, dv/dx = theta_z）
    uy = H1 * u1y + H2 * rz1 + H3 * u2y + H4 * rz2
    # 局所z方向のたわみ（回転は theta_y, dw/dx = -theta_y）
    uz = H1 * u1z - H2 * ry1 + H3 * u2z - H4 * ry2

    # 各 xi での無変形位置（全体）
    base = p1[None, :] + np.outer(xi, (p2 - p1))
    # 局所変位を全体へ（R^T @ d_local）
    d_local_curve = np.vstack([ux, uy, uz])  # (3, n)
    d_global = (R.T @ d_local_curve).T  # (n, 3)

    return base + scale * d_global


# ----------------------------------------------------------------------
# 2D / 3D 判定とプロット基盤
# ----------------------------------------------------------------------
def _is_planar(model: Model, result: StaticResult | None = None) -> bool:
    """x-y 平面問題とみなせるか判定する。"""
    if model.n_nodes == 0:
        return True
    if np.any(np.abs(model.nodes[:, 2]) > 1e-12):
        return False
    if result is not None:
        oop = []
        for i in range(model.n_nodes):
            d = result.node_disp(i)
            oop.extend([d[UZ], d[RX], d[RY]])
        if np.max(np.abs(oop)) > 1e-12:
            return False
    return True


def _new_axes(planar: bool):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    if planar:
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    return fig, ax


def _plot_line(ax, pts: np.ndarray, planar: bool, **kw):
    if planar:
        ax.plot(pts[:, 0], pts[:, 1], **kw)
    else:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **kw)


def _model_size(model: Model) -> float:
    if model.n_nodes < 2:
        return 1.0
    span = model.nodes.max(axis=0) - model.nodes.min(axis=0)
    return float(np.linalg.norm(span)) or 1.0


def _max_disp(model: Model, result: StaticResult) -> float:
    """並進変位の最大ノルム。"""
    mx = 0.0
    for i in range(model.n_nodes):
        d = result.node_disp(i)[:3]
        mx = max(mx, float(np.linalg.norm(d)))
    return mx


# ----------------------------------------------------------------------
# 公開 API
# ----------------------------------------------------------------------
def plot_model(
    model: Model,
    show_nodes: bool = True,
    show_supports: bool = True,
    show_loads: bool = True,
    ax=None,
):
    """変形前のモデルを描画して (fig, ax) を返す。"""
    planar = _is_planar(model)
    if ax is None:
        fig, ax = _new_axes(planar)
    else:
        fig = ax.figure

    tris = [np.vstack([model.nodes[s.n1], model.nodes[s.n2], model.nodes[s.n3]])
            for s in model.shells]
    _draw_shell_faces(ax, tris, planar, facecolor="C0", edgecolor="0.4",
                      alpha=0.25, lw=0.8, zorder=0)

    for e in model.elements:
        pts = np.vstack([model.nodes[e.n1], model.nodes[e.n2]])
        _plot_line(ax, pts, planar, color="0.4", lw=1.5, zorder=1)

    if show_nodes:
        nd = model.nodes
        if planar:
            ax.scatter(nd[:, 0], nd[:, 1], s=18, color="k", zorder=3)
        else:
            ax.scatter(nd[:, 0], nd[:, 1], nd[:, 2], s=18, color="k", zorder=3)

    if show_supports:
        _draw_supports(ax, model, planar)
    if show_loads:
        _draw_loads(ax, model, planar)

    ax.set_title("Model")
    return fig, ax


def plot_deformed(
    model: Model,
    result: StaticResult,
    scale: float | str = "auto",
    n: int = 16,
    show_undeformed: bool = True,
    ax=None,
):
    """変形図を描画して (fig, ax) を返す。

    scale : 変形の拡大率。"auto" でモデル寸法の約 5% になるよう自動調整。
    n     : 要素ごとの補間点数。
    """
    planar = _is_planar(model, result)
    if ax is None:
        fig, ax = _new_axes(planar)
    else:
        fig = ax.figure

    if scale == "auto":
        md = _max_disp(model, result)
        scale = (0.05 * _model_size(model) / md) if md > 0 else 1.0

    dof_maps = element_dof_map(model)
    for e, dofs in zip(model.elements, dof_maps):
        p1, p2 = model.nodes[e.n1], model.nodes[e.n2]
        if show_undeformed:
            _plot_line(ax, np.vstack([p1, p2]), planar, color="0.75", lw=1.0, ls="--", zorder=1)
        u_elem = result.u[dofs]
        if isinstance(e, TrussElement):
            xi = np.linspace(0.0, 1.0, n)[:, None]
            curve = ((1.0 - xi) * p1 + xi * p2
                     + scale * ((1.0 - xi) * u_elem[:3] + xi * u_elem[6:9]))
        else:
            curve = element_deformed_curve(p1, p2, u_elem, e.vref, scale=scale, n=n)
        _plot_line(ax, curve, planar, color="C0", lw=1.8, zorder=2)

    # シェル要素は節点並進変位で面を移動して描く（節点間は平面補間）
    if model.shells:
        und, defm = [], []
        for s in model.shells:
            verts = np.vstack([model.nodes[s.n1], model.nodes[s.n2], model.nodes[s.n3]])
            disp = np.vstack([result.node_disp(s.n1)[:3],
                              result.node_disp(s.n2)[:3],
                              result.node_disp(s.n3)[:3]])
            und.append(verts)
            defm.append(verts + scale * disp)
        if show_undeformed:
            for v in und:  # 変形前は辺のワイヤフレームで薄く描く
                _plot_line(ax, np.vstack([v, v[0]]), planar, color="0.75",
                           lw=0.8, ls="--", zorder=1)
        _draw_shell_faces(ax, defm, planar, facecolor="C0", edgecolor="0.3",
                          alpha=0.35, lw=0.8, zorder=2)

    ax.set_title(f"Deformed shape (x{scale:.3g})")
    return fig, ax


# ----------------------------------------------------------------------
# 支持・荷重のマーカー
# ----------------------------------------------------------------------
def _node_constraint_dofs(model: Model, node: int) -> set[int]:
    return {d for (nd, d), _ in model.constraints.items() if nd == node}


def _draw_supports(ax, model: Model, planar: bool):
    s = 0.025 * _model_size(model)
    for i in range(model.n_nodes):
        dofs = _node_constraint_dofs(model, i)
        # 面内自由度(UX,UY,RZ)に着目して支持種別を簡易表示
        trans = {UX, UY} & dofs
        if not trans:
            continue
        p = model.nodes[i]
        fixed_rot = RZ in dofs
        marker = "s" if fixed_rot else "^"  # 固定:四角, ピン:三角
        color = "tab:red"
        if planar:
            ax.scatter(p[0], p[1], marker=marker, s=90, c=color, zorder=4)
        else:
            ax.scatter(p[0], p[1], p[2], marker=marker, s=90, c=color, zorder=4)


def _draw_loads(ax, model: Model, planar: bool):
    if not model.nodal_loads:
        return
    # 力の最大値で矢印長を正規化
    forces = {}
    for (node, dof), val in model.nodal_loads.items():
        if dof in (UX, UY, UZ):
            forces.setdefault(node, np.zeros(3))[dof] += val
    if not forces:
        return
    fmax = max((np.linalg.norm(v) for v in forces.values()), default=1.0) or 1.0
    alen = 0.12 * _model_size(model)
    for node, fvec in forces.items():
        p = model.nodes[node]
        d = fvec / fmax * alen
        if planar:
            ax.annotate(
                "",
                xy=(p[0] + d[0], p[1] + d[1]),
                xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2),
                zorder=5,
            )
        else:
            ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], color="tab:green", lw=2)


def plot_diagram(
    forces,
    component: str,
    scale: float | str = "auto",
    n: int = 12,
    fill: bool = True,
    ax=None,
):
    """断面力図（指定した1成分）を描画して (fig, ax) を返す。

    forces    : ForceResults（recover_forces の戻り値）
    component : "N","Vy","Vz","T","My","Mz" のいずれか
    scale     : 値→長さの倍率。"auto" でモデル寸法の約12%に正規化。

    各部材の材軸に直交方向へ値をオフセットして描く。Mz/Vy は局所y、
    My/Vz は局所z、N/T は局所y方向にプロットする。
    """
    model = forces.model
    planar = _is_planar(model)
    if ax is None:
        fig, ax = _new_axes(planar)
    else:
        fig = ax.figure

    # オフセット方向（局所軸の選択）: 0=e1(x),1=e2(y),2=e3(z)
    dir_idx = {"Mz": 1, "Vy": 1, "N": 1, "T": 1, "My": 2, "Vz": 2}.get(component, 1)

    # 自動スケール
    if scale == "auto":
        vmax = max((ef.max_abs(component) for ef in forces.elements), default=0.0)
        scale = (0.12 * _model_size(model) / vmax) if vmax > 0 else 1.0

    xi = np.linspace(0.0, 1.0, n)
    for e, ef in zip(model.elements, forces.elements):
        p1, p2 = model.nodes[e.n1], model.nodes[e.n2]
        R = rotation_matrix(p1, p2, None if isinstance(e, TrussElement) else e.vref)
        d = R[dir_idx]  # 全体座標でのオフセット方向
        base = p1[None, :] + np.outer(xi, (p2 - p1))
        vals = ef.value(component, xi)
        offset = base + scale * np.outer(vals, d)

        # 部材線
        _plot_line(ax, np.vstack([p1, p2]), planar, color="0.5", lw=1.2, zorder=1)
        # 図形
        if fill and planar:
            xs = np.concatenate([base[:, 0], offset[::-1, 0]])
            ys = np.concatenate([base[:, 1], offset[::-1, 1]])
            ax.fill(xs, ys, color="C3", alpha=0.25, zorder=2)
            # 縦ハッチ（値の縦線）
            for j in range(n):
                ax.plot([base[j, 0], offset[j, 0]], [base[j, 1], offset[j, 1]],
                        color="C3", lw=0.5, alpha=0.5, zorder=2)
        _plot_line(ax, offset, planar, color="C3", lw=1.6, zorder=3)

    ax.set_title(f"{component} diagram (x{scale:.3g})")
    return fig, ax


def _draw_sized_members(
    nodes, pairs, values, planar, max_width, min_width, cmap, label,
    show_colorbar, ax, fig,
):
    """節点座標・部材ペア・値から、線幅/色を値に比例させて描く共通処理。"""
    import matplotlib.cm as cm
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    values = np.asarray(values, dtype=float)
    vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin
    norm = Normalize(vmin=vmin, vmax=vmax if span > 0 else vmin + 1.0)
    colormap = colormaps[cmap]

    for (i, j), v in zip(pairs, values):
        pts = np.vstack([nodes[i], nodes[j]])
        frac = (v - vmin) / span if span > 0 else 1.0
        lw = min_width + frac * (max_width - min_width)
        _plot_line(ax, pts, planar, color=colormap(norm(v)), lw=lw,
                   solid_capstyle="round", zorder=2)

    if show_colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        if label:
            cb.set_label(label)


def plot_member_sizes(
    model: Model,
    values,
    max_width: float = 9.0,
    min_width: float = 0.8,
    cmap: str = "viridis",
    label: str = "",
    show_colorbar: bool = True,
    show_undeformed: bool = False,
    ax=None,
):
    """各部材を、与えた値に比例した線幅・色で描く（構造形態の図示）。

    サイジング最適化結果の「どの部材が太く/細くなったか」を可視化する用途。
    values は要素数と同じ長さの配列（断面スケール・断面積・代表寸法など）。
    線幅は値に比例、色も値で着色する。2D/3D を自動判定する。
    """
    values = np.asarray(values, dtype=float)
    if len(values) != len(model.elements):
        raise ValueError("values の長さが要素数と一致しません")

    planar = _is_planar(model)
    if ax is None:
        fig, ax = _new_axes(planar)
    else:
        fig = ax.figure

    if show_undeformed:
        for el in model.elements:
            _plot_line(ax, np.vstack([model.nodes[el.n1], model.nodes[el.n2]]),
                       planar, color="0.85", lw=0.8, zorder=1)

    pairs = [(el.n1, el.n2) for el in model.elements]
    _draw_sized_members(model.nodes, pairs, values, planar, max_width, min_width,
                        cmap, label, show_colorbar, ax, fig)
    ax.set_title(label or "Member sizes")
    return fig, ax


def plot_truss(
    nodes,
    members,
    areas,
    rel_tol: float = 1e-3,
    show_all: bool = False,
    max_width: float = 9.0,
    min_width: float = 1.0,
    cmap: str = "viridis",
    label: str = "area",
    show_colorbar: bool = True,
    ax=None,
):
    """トラス（地盤構造）の最適配置を図示する。

    トポロジー最適化の結果（断面積 areas）を、線幅・色を断面積に比例させて描く。
    既定では断面積が最大の rel_tol 倍以下の部材（≈除去された部材）を描かない。
    show_all=True で全候補部材を薄く重ねて表示する。
    """
    nodes = np.asarray(nodes, dtype=float)
    areas = np.asarray(areas, dtype=float)
    planar = nodes.shape[1] == 2
    if ax is None:
        fig, ax = _new_axes(planar)
    else:
        fig = ax.figure

    amax = areas.max() if areas.size else 0.0
    keep = areas > rel_tol * amax if amax > 0 else np.zeros(len(areas), dtype=bool)

    if show_all:
        for (i, j) in members:
            _plot_line(ax, np.vstack([nodes[i], nodes[j]]), planar,
                       color="0.88", lw=0.6, zorder=1)

    pairs = [members[e] for e in range(len(members)) if keep[e]]
    vals = areas[keep]
    if len(pairs):
        _draw_sized_members(nodes, pairs, vals, planar, max_width, min_width,
                            cmap, label, show_colorbar, ax, fig)
    # 節点
    if planar:
        ax.scatter(nodes[:, 0], nodes[:, 1], s=8, color="k", zorder=3)
    else:
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=8, color="k", zorder=3)
    ax.set_title(f"Optimal layout ({len(pairs)}/{len(members)} members)")
    return fig, ax


def show():
    """matplotlib のウィンドウを表示する（plt.show のラッパ）。"""
    import matplotlib.pyplot as plt

    plt.show()


def savefig(path: str, **kw) -> str:
    """図を保存する。相対パスは workspace フォルダ内に保存され、保存先を返す。"""
    import matplotlib.pyplot as plt

    from .workspace import resolve

    full = resolve(path)
    plt.savefig(full, **kw)
    return full
