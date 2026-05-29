"""サイジング最適化の駆動ループ（MMA を反復適用）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mma import mmasub
from .sizing import SizingProblem


@dataclass
class OptResult:
    """最適化結果。"""

    x: np.ndarray            # 最適設計変数
    mass: float              # 最適質量
    constraints: np.ndarray  # 制約値（≤0 で満足）
    sections: dict           # 各設計グループの最適断面
    iterations: int
    converged: bool
    history: list            # [(iter, mass, max_constraint, change), ...]


def minimize_mass(
    problem: SizingProblem,
    x0=None,
    maxiter: int = 100,
    move: float = 0.2,
    tol: float = 1e-4,
    c_mma: float = 1e4,
    verbose: bool = False,
) -> OptResult:
    """質量最小化をMMAで解く。

    problem : SizingProblem（目的・制約・解析的感度を提供）
    move    : MMA の移動制限（小さいほど慎重）
    tol     : 設計変数の変化量がこれを下回ったら収束
    """
    n = problem.n_var
    xmin, xmax = problem.bounds()
    x = problem.x0() if x0 is None else np.asarray(x0, dtype=float)
    x = np.clip(x, xmin, xmax)

    xval = x.reshape(-1, 1)
    xminc = xmin.reshape(-1, 1)
    xmaxc = xmax.reshape(-1, 1)
    xold1 = xval.copy()
    xold2 = xval.copy()
    low = xminc.copy()
    upp = xmaxc.copy()

    f0, df0, fval, dfdx = problem.evaluate(xval.flatten())
    m = len(fval)
    a0 = 1.0
    a = np.zeros((m, 1))
    c = c_mma * np.ones((m, 1))
    d = np.zeros((m, 1))

    history = []
    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        f0, df0, fval, dfdx = problem.evaluate(xval.flatten())
        f0c = np.array([[f0]])
        df0c = df0.reshape(-1, 1)
        fvalc = fval.reshape(-1, 1)
        dfdxc = dfdx.reshape(m, n)

        xmma, low, upp = mmasub(
            m, n, it, xval, xminc, xmaxc, xold1, xold2,
            f0c, df0c, fvalc, dfdxc, low, upp, a0, a, c, d, move,
        )
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()

        change = float(np.max(np.abs(xval - xold1)))
        gmax = float(fval.max()) if m else 0.0
        history.append((it, f0, gmax, change))
        if verbose:
            print(f"  it{it:3d}  mass={f0:12.4f}  max_g={gmax:+.3e}  change={change:.3e}")
        if change < tol and it > 2:
            converged = True
            break

    xfin = xval.flatten()
    f0, df0, fval, dfdx = problem.evaluate(xfin)
    return OptResult(
        x=xfin,
        mass=f0,
        constraints=fval,
        sections=problem.current_sections(xfin),
        iterations=it,
        converged=converged,
        history=history,
    )
