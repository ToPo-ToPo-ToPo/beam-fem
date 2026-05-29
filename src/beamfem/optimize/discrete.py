"""離散サイジング最適化（規格サイズのカタログから選ぶ組合せ最適化）。

各設計変数（断面スケール係数）を、与えた離散カタログの値からのみ選び、応力・たわみ
制約のもとで総質量を最小化する。連続版（[`sizing.py`](sizing.py)）と同じ
`SizingProblem` を関数評価器として用いる。

2 つの解法：

- ``solve_discrete_exhaustive`` : 全組合せを評価（小規模で**大域最適**、検証用）
- ``solve_discrete_greedy``     : 連続最適解から丸めて開始→実行可能化→近傍局所探索
                                  （実用規模に対応する発見的手法）
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .driver import minimize_mass
from .sizing import SizingProblem


@dataclass
class DiscreteResult:
    """離散最適化の結果。"""

    x: np.ndarray            # 選ばれたスケール値（各設計変数）
    indices: list[int]       # カタログ内インデックス
    mass: float
    constraints: np.ndarray  # 制約値（≤0 で満足）
    feasible: bool
    n_eval: int              # 関数評価回数
    method: str


def _normalize_catalogs(catalogs, n: int):
    """カタログを「各変数ごとの昇順 1 次元配列のリスト」に正規化する。

    要素がスカラーなら単一カタログを全変数で共有、配列なら各変数ごとと解釈する。
    """
    if np.isscalar(catalogs[0]):
        single = np.sort(np.asarray(catalogs, dtype=float))
        return [single.copy() for _ in range(n)]
    if len(catalogs) != n:
        raise ValueError(f"カタログ数 {len(catalogs)} が設計変数 {n} と一致しません")
    return [np.sort(np.asarray(c, dtype=float)) for c in catalogs]


def _round_up_index(cat: np.ndarray, value: float) -> int:
    """value 以上で最小のカタログ値のインデックス（無ければ最大値）。"""
    idx = np.searchsorted(cat, value, side="left")
    return int(min(idx, len(cat) - 1))


def solve_discrete_exhaustive(
    problem: SizingProblem,
    catalogs,
    tol: float = 1e-6,
    max_combos: int = 200_000,
) -> DiscreteResult:
    """全組合せを評価して最小質量の実行可能解を返す（大域最適）。

    組合せ数が max_combos を超える場合はエラー（greedy を使う）。
    """
    n = problem.n_var
    cats = _normalize_catalogs(catalogs, n)
    total = int(np.prod([len(c) for c in cats]))
    if total > max_combos:
        raise ValueError(
            f"組合せ数 {total} が上限 {max_combos} を超えます。solve_discrete_greedy を使ってください。"
        )

    best = None
    n_eval = 0
    for combo in itertools.product(*[range(len(c)) for c in cats]):
        x = np.array([cats[i][combo[i]] for i in range(n)])
        f0, fv = problem.evaluate_values(x)
        n_eval += 1
        feasible = fv.size == 0 or fv.max() <= tol
        if feasible and (best is None or f0 < best[0]):
            best = (f0, list(combo), x, fv)

    if best is None:
        raise RuntimeError("実行可能な組合せがありません（カタログの上限を上げてください）。")
    f0, idx, x, fv = best
    problem.evaluate_values(x)  # モデルを最適点の断面に確定
    return DiscreteResult(x=x, indices=idx, mass=f0, constraints=fv,
                          feasible=True, n_eval=n_eval, method="exhaustive")


def solve_discrete_greedy(
    problem: SizingProblem,
    catalogs,
    warm_start: str = "continuous",
    tol: float = 1e-6,
    pairwise: bool = True,
    verbose: bool = False,
) -> DiscreteResult:
    """連続解から丸めて開始し、実行可能化＋近傍局所探索で離散最適化する。

    warm_start : "continuous"（連続最適解を丸める）/ "max"（最大サイズから開始）
    pairwise   : True で「一方を太く・他方を細く」の交換移動も探索（材料再配分）
    """
    n = problem.n_var
    cats = _normalize_catalogs(catalogs, n)
    sizes = [len(c) for c in cats]
    n_eval = 0

    def feval(idx):
        nonlocal n_eval
        x = np.array([cats[i][idx[i]] for i in range(n)])
        f0, fv = problem.evaluate_values(x)
        n_eval += 1
        gmax = fv.max() if fv.size else -np.inf
        return f0, fv, gmax

    # --- 初期点 ---
    if warm_start == "continuous":
        cont = minimize_mass(problem, maxiter=80, move=0.2, tol=1e-5)
        idx = [_round_up_index(cats[i], cont.x[i]) for i in range(n)]
    else:
        idx = [sizes[i] - 1 for i in range(n)]
    f0, fv, gmax = feval(idx)

    # --- 実行可能化（最も制約違反を減らす増加を貪欲に） ---
    repair_cap = sum(sizes) + n
    steps = 0
    while gmax > tol and steps < repair_cap:
        steps += 1
        best = None
        for i in range(n):
            if idx[i] < sizes[i] - 1:
                j = idx.copy(); j[i] += 1
                _, _, gj = feval(j)
                if best is None or gj < best[1]:
                    best = (j, gj)
        if best is None or best[1] >= gmax - 1e-15:
            break  # これ以上改善できない（全て最大 or 進展なし）
        idx, gmax = best[0], best[1]
        f0, fv, gmax = feval(idx)
    feasible = gmax <= tol

    # --- 質量削減の局所探索（近傍: 単変数縮小 + 任意で交換移動） ---
    improved = True
    while improved and feasible:
        improved = False
        best_move = None  # (idx, f0, fv)
        # 単変数を 1 段細く
        for i in range(n):
            if idx[i] > 0:
                j = idx.copy(); j[i] -= 1
                fj, fvj, gj = feval(j)
                if gj <= tol and fj < f0 - 1e-12:
                    if best_move is None or fj < best_move[1]:
                        best_move = (j, fj, fvj)
        # 交換: 一方を太く・他方を細く（質量が減る組合せ）
        if pairwise:
            for i in range(n):
                if idx[i] >= sizes[i] - 1:
                    continue
                for k in range(n):
                    if k == i or idx[k] <= 0:
                        continue
                    j = idx.copy(); j[i] += 1; j[k] -= 1
                    fj, fvj, gj = feval(j)
                    if gj <= tol and fj < f0 - 1e-12:
                        if best_move is None or fj < best_move[1]:
                            best_move = (j, fj, fvj)
        if best_move is not None:
            idx, f0, fv = best_move
            improved = True
            if verbose:
                print(f"  [greedy] mass={f0:.4f}  idx={idx}")

    x = np.array([cats[i][idx[i]] for i in range(n)])
    problem.evaluate_values(x)  # モデルを最適点の断面に確定（後続の解析と整合）
    return DiscreteResult(x=x, indices=list(idx), mass=f0, constraints=fv,
                          feasible=feasible, n_eval=n_eval, method="greedy")
