import numpy as np
from src.math_utils import njit, wrap_rad, RAD2DEG

@njit(fastmath=True)
def jit_compute_pa_and_lse_sparse(
    z: np.ndarray, w_inv_sqrt: np.ndarray, hw_indptr: np.ndarray, hw_indices: np.ndarray,
    hw_data: np.ndarray, rh_indptr: np.ndarray, rh_indices: np.ndarray, rh_data: np.ndarray,
    rh_diag: np.ndarray, r_indptr: np.ndarray, r_indices: np.ndarray, r_data: np.ndarray,
    r_diag: np.ndarray, ws_b: np.ndarray, ws_u: np.ndarray, v_lse_out: np.ndarray
) -> float:
    m = len(z)
    nb = len(rh_diag)
    norm_zw_sq = 0.0
    for k in range(m):
        zr = z[k].real * w_inv_sqrt[k]
        zi = z[k].imag * w_inv_sqrt[k]
        norm_zw_sq += zr * zr + zi * zi

    for i in range(nb):
        acc = 0.0 + 0.0j
        for p in range(hw_indptr[i], hw_indptr[i + 1]):
            acc += hw_data[p] * z[hw_indices[p]]
        ws_b[i] = acc

    norm_u_sq = 0.0
    for i in range(nb):
        s = ws_b[i]
        for p in range(rh_indptr[i], rh_indptr[i + 1]):
            s -= rh_data[p] * ws_u[rh_indices[p]]
        ui = s / rh_diag[i]
        ws_u[i] = ui
        norm_u_sq += ui.real * ui.real + ui.imag * ui.imag

    for i in range(nb - 1, -1, -1):
        s = ws_u[i]
        for p in range(r_indptr[i], r_indptr[i + 1]):
            s -= r_data[p] * v_lse_out[r_indices[p]]
        v_lse_out[i] = s / r_diag[i]

    cos_theta = np.sqrt(norm_u_sq / norm_zw_sq)
    if cos_theta > 1.0: cos_theta = 1.0
    elif cos_theta < -1.0: cos_theta = -1.0
    return np.arccos(cos_theta) * RAD2DEG

@njit(fastmath=True)
def jit_sparse_spd_cholesky_solve(
    JTJ: np.ndarray, JTr: np.ndarray, dx: np.ndarray,
    col_nz_idx: np.ndarray, col_nz_cnt: np.ndarray, n_states: int
) -> None:
    for k in range(n_states):
        diag_val = JTJ[k, k]
        if diag_val <= 1e-12: diag_val = 1e-12
        l_kk = np.sqrt(diag_val)
        JTJ[k, k] = l_kk
        inv_lkk = 1.0 / l_kk
        cnt = col_nz_cnt[k]
        for a in range(cnt):
            r = col_nz_idx[k, a]
            l_rk = JTJ[r, k] * inv_lkk
            JTJ[r, k] = l_rk
            for b in range(a + 1):
                c = col_nz_idx[k, b]
                JTJ[r, c] -= l_rk * JTJ[c, k]

    for k in range(n_states): dx[k] = JTr[k]
    for k in range(n_states):
        yk = dx[k] / JTJ[k, k]
        dx[k] = yk
        cnt = col_nz_cnt[k]
        for a in range(cnt):
            r = col_nz_idx[k, a]
            dx[r] -= JTJ[r, k] * yk

    for k in range(n_states - 1, -1, -1):
        s = dx[k]
        cnt = col_nz_cnt[k]
        for a in range(cnt):
            r = col_nz_idx[k, a]
            s -= JTJ[r, k] * dx[r]
        dx[k] = s / JTJ[k, k]

@njit(fastmath=True)
def jit_solve_scalar_nls_sparse(
    vm_target: np.ndarray, p_target: np.ndarray, q_target: np.ndarray,
    w_vm_eff_base: np.ndarray, w_pq_eff_base: np.ndarray, pmu_buses: np.ndarray,
    bi_arr: np.ndarray, bj_arr: np.ndarray, ys_re: np.ndarray, ys_im: np.ndarray,
    ym_re: np.ndarray, ym_im: np.ndarray, v_warm: np.ndarray, nb: int,
    robust_huber: bool, max_iter: int, inv_perm: np.ndarray,
    ws_x_curr: np.ndarray, ws_w_irls_sq: np.ndarray,
    ws_va: np.ndarray, ws_vm: np.ndarray, ws_v_re: np.ndarray, ws_v_im: np.ndarray,
    ws_JTJ: np.ndarray, ws_JTr: np.ndarray, ws_dx: np.ndarray, ws_col_nz_idx: np.ndarray,
    ws_col_nz_cnt: np.ndarray, ws_idx_u: np.ndarray, ws_jp_u: np.ndarray, ws_jq_u: np.ndarray,
    v_out: np.ndarray
) -> None:
    n_v = len(vm_target)
    n_i = len(p_target)
    n_meas = n_v + 2 * n_i
    n_states = 2 * nb - 1
    huber_c = 1.6

    va0 = np.angle(v_warm[0])
    for k in range(nb - 1): ws_x_curr[k] = wrap_rad(np.angle(v_warm[k + 1]) - va0)
    for k in range(nb): ws_x_curr[(nb - 1) + k] = np.abs(v_warm[k])

    for m_idx in range(n_meas): ws_w_irls_sq[m_idx] = 1.0

    for _ in range(max_iter):
        ws_va[0] = 0.0
        ws_vm[0] = ws_x_curr[nb - 1]
        ws_v_re[0] = ws_vm[0]
        ws_v_im[0] = 0.0
        for k in range(1, nb):
            ang = ws_x_curr[k - 1]
            mag = ws_x_curr[(nb - 1) + k]
            ws_va[k] = ang
            ws_vm[k] = mag
            ws_v_re[k] = mag * np.cos(ang)
            ws_v_im[k] = mag * np.sin(ang)

        for k in range(n_states):
            ws_JTr[k] = 0.0
            ws_JTJ[k, k] = 1e-9
            cnt = ws_col_nz_cnt[k]
            for a in range(cnt):
                ws_JTJ[ws_col_nz_idx[k, a], k] = 0.0

        for k in range(n_v):
            b_idx = pmu_buses[k]
            v_col = inv_perm[(nb - 1) + b_idx]
            wb = w_vm_eff_base[k]
            w_sq = (wb * wb) * ws_w_irls_sq[k]
            diff_v = ws_vm[b_idx] - vm_target[k]

            ws_JTJ[v_col, v_col] += w_sq
            ws_JTr[v_col] -= w_sq * diff_v

            if robust_huber:
                ar = abs(diff_v * wb)
                if ar <= huber_c: ws_w_irls_sq[k] = 1.0
                else: ws_w_irls_sq[k] = huber_c / (ar + 1e-9)

        for k in range(n_i):
            bi, bj = bi_arr[k], bj_arr[k]
            vi_r, vi_i = ws_v_re[bi], ws_v_im[bi]
            vj_r, vj_i = ws_v_re[bj], ws_v_im[bj]
            vmi, vmj = ws_vm[bi], ws_vm[bj]
            vivj_r = vi_r * vj_r + vi_i * vj_i
            vivj_i = vi_i * vj_r - vi_r * vj_i
            tm_r = vivj_r * ym_re[k] + vivj_i * ym_im[k]
            tm_i = vivj_i * ym_re[k] - vivj_r * ym_im[k]
            vmi_sq = vmi * vmi
            dp = (vmi_sq * ys_re[k] + tm_r) - p_target[k]
            dq = (-vmi_sq * ys_im[k] + tm_i) - q_target[k]

            rp_idx = n_v + k
            rq_idx = n_v + n_i + k
            wb = w_pq_eff_base[k]
            wb_sq = wb * wb
            wp_sq = wb_sq * ws_w_irls_sq[rp_idx]
            wq_sq = wb_sq * ws_w_irls_sq[rq_idx]

            if robust_huber:
                arp, arq = abs(dp * wb), abs(dq * wb)
                ws_w_irls_sq[rp_idx] = 1.0 if arp <= huber_c else huber_c / (arp + 1e-9)
                ws_w_irls_sq[rq_idx] = 1.0 if arq <= huber_c else huber_c / (arq + 1e-9)

            inv_vmi, inv_vmj = 1.0 / vmi, 1.0 / vmj
            dP_dvmi = 2.0 * vmi * ys_re[k] + tm_r * inv_vmi
            dQ_dvmi = -2.0 * vmi * ys_im[k] + tm_i * inv_vmi
            dP_dvmj = tm_r * inv_vmj
            dQ_dvmj = tm_i * inv_vmj

            n_loc = 0
            if bi > 0:
                ws_idx_u[n_loc], ws_jp_u[n_loc], ws_jq_u[n_loc] = inv_perm[bi - 1], -tm_i, tm_r
                n_loc += 1
            if bj > 0:
                ws_idx_u[n_loc], ws_jp_u[n_loc], ws_jq_u[n_loc] = inv_perm[bj - 1], tm_i, -tm_r
                n_loc += 1

            ws_idx_u[n_loc], ws_jp_u[n_loc], ws_jq_u[n_loc] = inv_perm[(nb - 1) + bi], dP_dvmi, dQ_dvmi
            n_loc += 1
            ws_idx_u[n_loc], ws_jp_u[n_loc], ws_jq_u[n_loc] = inv_perm[(nb - 1) + bj], dP_dvmj, dQ_dvmj
            n_loc += 1

            wp_dp, wq_dq = wp_sq * dp, wq_sq * dq
            for a in range(n_loc):
                ua, jpa, jqa = ws_idx_u[a], ws_jp_u[a], ws_jq_u[a]
                ws_JTr[ua] -= wp_dp * jpa + wq_dq * jqa
                for b in range(a + 1):
                    ub = ws_idx_u[b]
                    val = wp_sq * jpa * ws_jp_u[b] + wq_sq * jqa * ws_jq_u[b]
                    if ua >= ub: ws_JTJ[ua, ub] += val
                    else: ws_JTJ[ub, ua] += val

        jit_sparse_spd_cholesky_solve(ws_JTJ, ws_JTr, ws_dx, ws_col_nz_idx, ws_col_nz_cnt, n_states)

        max_dx = 0.0
        for s_orig in range(n_states):
            step_val = ws_dx[inv_perm[s_orig]]
            ws_x_curr[s_orig] += step_val
            abs_step = abs(step_val)
            if abs_step > max_dx: max_dx = abs_step

        if max_dx < 1e-6: break

    v_out[0] = ws_x_curr[nb - 1] + 0.0j
    for k in range(1, nb):
        ang = ws_x_curr[k - 1]
        mag = ws_x_curr[(nb - 1) + k]
        v_out[k] = mag * (np.cos(ang) + 1j * np.sin(ang))
