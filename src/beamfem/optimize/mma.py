"""Method of Moving Asymptotes (MMA, Svanberg) の実装。

構造最適化の標準アルゴリズム。各反復で移動漸近線による分離可能・凸な部分問題を
構成し、主双対内点法（subsolv）で解く。Svanberg(1987/2007) の mmasub/subsolv の
忠実な実装。

最小化問題::

    min  f0(x) + a0*z + Σ (c_i y_i + 0.5 d_i y_i^2)
    s.t. f_i(x) - a_i z - y_i ≤ 0
         xmin ≤ x ≤ xmax,  y ≥ 0,  z ≥ 0

既定 (a0=1, a=0, c=大, d=0) で通常の制約付き最小化 min f0 s.t. f_i≤0 になる。
"""

from __future__ import annotations

import numpy as np

EPSIMIN = 1e-7


def mmasub(
    m, n, it, xval, xmin, xmax, xold1, xold2,
    f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move,
):
    """MMA 部分問題を構成して解き、(xmma, low, upp) を返す。"""
    raa0 = 1e-5
    albefa = 0.1
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7
    een = np.ones((n, 1))
    eem = np.ones((m, 1))

    # --- 移動漸近線 low, upp ---
    if it <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = een.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 10.0 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 10.0 * (xmax - xmin)
        low = np.maximum(np.minimum(low, lowmax), lowmin)
        upp = np.minimum(np.maximum(upp, uppmin), uppmax)

    # --- 試行領域 alfa, beta ---
    zzz = np.maximum(low + albefa * (xval - low), xval - move * (xmax - xmin))
    alfa = np.maximum(zzz, xmin)
    zzz = np.minimum(upp - albefa * (upp - xval), xval + move * (xmax - xmin))
    beta = np.minimum(zzz, xmax)

    # --- p0,q0 / P,Q ---
    xmami = np.maximum(xmax - xmin, 1e-5 * een)
    xmamiinv = een / xmami
    ux1 = upp - xval
    xl1 = xval - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1

    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    P = np.maximum(dfdx, 0.0)
    Q = np.maximum(-dfdx, 0.0)
    PQ = 0.001 * (P + Q) + raa0 * (eem @ xmamiinv.T)
    P = (P + PQ) * ux2.flatten()[None, :]
    Q = (Q + PQ) * xl2.flatten()[None, :]

    b = P @ (een / ux1) + Q @ (een / xl1) - fval

    xmma = subsolv(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    return xmma, low, upp


def subsolv(m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    """MMA 部分問題を主双対内点法で解き、最適 x を返す。"""
    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1.0
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.ones((1, 1))
    lam = eem.copy()
    xsi = np.maximum(een / (x - alfa), een)
    eta = np.maximum(een / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.ones((1, 1))
    s = eem.copy()

    while epsi > EPSIMIN:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ (een / ux1) + Q @ (een / xl1)
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu = np.concatenate(
            (rex, rey, rez, relam, rexsi, reeta, remu, rezet, res), axis=0
        )
        residunorm = np.sqrt((residu.T @ residu).item())
        residumax = np.max(np.abs(residu))

        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ (een / ux1) + Q @ (een / xl1)
            GG = P * uxinv2.flatten()[None, :] - Q * xlinv2.flatten()[None, :]
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - a.T @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2.0 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                bb = np.concatenate((blam, delz), axis=0)
                Alam = np.diagflat(diaglamyi) + (GG * diagxinv.flatten()[None, :]) @ GG.T
                AA = np.block([[Alam, a], [a.T, -zet / z]])
                solut = np.linalg.solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m:m + 1]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = np.diagflat(diagx) + (GG.T * diaglamyiinv.flatten()[None, :]) @ GG
                azz = zet / z + a.T @ (a / diaglamyi)
                axz = -GG.T @ (a / diaglamyi)
                bx = delx + GG.T @ (dellamyi / diaglamyi)
                bz = delz - a.T @ (dellamyi / diaglamyi)
                AA = np.block([[Axx, axz], [axz.T, azz]])
                bb = np.concatenate((-bx, -bz), axis=0)
                solut = np.linalg.solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n:n + 1]
                dlam = GG @ dx / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam

            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)

            stmxx = np.max(-1.01 * dxx / xx)
            stmalfa = np.max(-1.01 * dx / (x - alfa))
            stmbeta = np.max(1.01 * dx / (beta - x))
            stminv = max(stmxx, stmalfa, stmbeta, 1.0)
            steg = 1.0 / stminv

            xold, yold, zold = x.copy(), y.copy(), z.copy()
            lamold, xsiold, etaold = lam.copy(), xsi.copy(), eta.copy()
            muold, zetold, sold = mu.copy(), zet.copy(), s.copy()

            resinew = 2.0 * residunorm
            itto = 0
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ (een / ux1) + Q @ (een / xl1)
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - a.T @ lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu = np.concatenate(
                    (rex, rey, rez, relam, rexsi, reeta, remu, rezet, res), axis=0
                )
                resinew = np.sqrt((residu.T @ residu).item())
                steg = steg / 2.0
            residunorm = resinew
            residumax = np.max(np.abs(residu))
        epsi = 0.1 * epsi

    return x
