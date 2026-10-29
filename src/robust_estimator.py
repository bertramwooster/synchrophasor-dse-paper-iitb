import numpy as np
from src.math_utils import njit, TWO_PI, RAD2DEG, DEG2RAD, wrap_rad, wrap_deg, fast_1d_sort_cluster_inplace, weighted_median_1d_subset, jit_median_1d
from src.sparse_solvers import jit_compute_pa_and_lse_sparse, jit_solve_scalar_nls_sparse

@njit(fastmath=True)
def jit_step_proposed_robust(
    z_t: np.ndarray, f_meas_t: np.ndarray, rocof_meas_t: np.ndarray, prop_init_flag: np.ndarray,
    prop_v_clean: np.ndarray, prop_va_pmu_clean: np.ndarray, prop_f_clean: np.ndarray,
    prop_anchor_va_pmu: np.ndarray, prop_anchor_state: np.ndarray,
    pmu_buses: np.ndarray, v_meas_idx: np.ndarray, i_meas_idx: np.ndarray,
    i_owner_pmu: np.ndarray, i_parent_v_idx: np.ndarray, bi_arr: np.ndarray,
    bj_arr: np.ndarray, ys_re: np.ndarray, ys_im: np.ndarray, ym_re: np.ndarray, ym_im: np.ndarray,
    w_vm_base: np.ndarray, w_pq_base: np.ndarray, w_inv_sqrt: np.ndarray,
    hw_indptr: np.ndarray, hw_indices: np.ndarray, hw_data: np.ndarray,
    rh_indptr: np.ndarray, rh_indices: np.ndarray, rh_data: np.ndarray, rh_diag: np.ndarray,
    r_indptr: np.ndarray, r_indices: np.ndarray, r_data: np.ndarray, r_diag: np.ndarray,
    pa_threshold_deg: float, f0: float, dt: float, nb: int, slack_idx: int, inv_perm: np.ndarray,
    ws_b: np.ndarray, ws_u: np.ndarray,
    ws_v_lse: np.ndarray, ws_v_rob: np.ndarray, ws_ang_pmu_raw: np.ndarray, ws_vm_pmu_raw: np.ndarray,
    ws_r_adc_fdia: np.ndarray, ws_r_dsp_f: np.ndarray, ws_w_vm_eff: np.ndarray,
    ws_w_pq_eff: np.ndarray, ws_vm_target: np.ndarray, ws_p_target: np.ndarray, ws_q_target: np.ndarray,
    ws_delta_diff_deg: np.ndarray, ws_trust_weights: np.ndarray, ws_labels: np.ndarray,
    ws_order: np.ndarray, ws_cluster_trust: np.ndarray, ws_sub_vals: np.ndarray,
    ws_sub_weights: np.ndarray, ws_med_buf: np.ndarray, ws_x_curr: np.ndarray,
    ws_w_irls_sq: np.ndarray, ws_va: np.ndarray, ws_vm: np.ndarray, ws_v_re: np.ndarray,
    ws_v_im: np.ndarray, ws_JTJ: np.ndarray, ws_JTr: np.ndarray, ws_dx: np.ndarray,
    ws_col_nz_idx: np.ndarray, ws_col_nz_cnt: np.ndarray, ws_idx_u: np.ndarray,
    ws_jp_u: np.ndarray, ws_jq_u: np.ndarray, v_aligned_out: np.ndarray, est_spoof_deg_out: np.ndarray
) -> None:
    n_pmu = len(pmu_buses)
    n_i = len(i_meas_idx)

    pa_deg = jit_compute_pa_and_lse_sparse(
        z_t, w_inv_sqrt, hw_indptr, hw_indices, hw_data, rh_indptr, rh_indices, rh_data, rh_diag,
        r_indptr, r_indices, r_data, r_diag, ws_b, ws_u, ws_v_lse
    )

    if prop_init_flag[0] == 0:
        for k in range(nb):
            prop_v_clean[k] = ws_v_lse[k]
            v_aligned_out[k] = ws_v_lse[k]
        for p_idx in range(n_pmu):
            ang_init = np.angle(ws_v_lse[pmu_buses[p_idx]])
            prop_va_pmu_clean[p_idx] = ang_init
            prop_anchor_va_pmu[p_idx] = ang_init
            prop_f_clean[p_idx] = f_meas_t[p_idx]
            est_spoof_deg_out[p_idx] = 0.0
        prop_anchor_state[0] = np.angle(ws_v_lse[slack_idx])
        prop_anchor_state[1] = 0.0
        prop_init_flag[0] = 1
        return

    f_coi_est = jit_median_1d(prop_f_clean, ws_med_buf, n_pmu)
    d_ang_coi = TWO_PI * (f_coi_est - f0) * dt
    rot_coi = np.cos(d_ang_coi) + 1j * np.sin(d_ang_coi)

    max_r_adc = 0.0
    max_kin_jump = 0.0
    inv_two_pi_dt = 1.0 / (TWO_PI * dt)

    for p_idx in range(n_pmu):
        zv = z_t[v_meas_idx[p_idx]]
        ang_raw = np.angle(zv)
        vm_raw = np.abs(zv)
        ws_ang_pmu_raw[p_idx] = ang_raw
        ws_vm_pmu_raw[p_idx] = vm_raw

        vm_clean_p = np.abs(prop_v_clean[pmu_buses[p_idx]])
        vm_jump = abs(vm_raw - vm_clean_p)
        if vm_jump > 0.045:
            r_adc = vm_jump * 10.0
            ws_r_adc_fdia[p_idx] = r_adc
            ws_w_vm_eff[p_idx] = w_vm_base[p_idx] * 0.15
        else:
            ws_r_adc_fdia[p_idx] = 0.0
            ws_w_vm_eff[p_idx] = w_vm_base[p_idx]

        d_theta_meas = wrap_rad(ang_raw - prop_va_pmu_clean[p_idx])
        f_from_phase = f0 + d_theta_meas * inv_two_pi_dt
        ws_r_dsp_f[p_idx] = abs(f_meas_t[p_idx] - f_from_phase)

        kin_jump_deg = abs(d_theta_meas - TWO_PI * (prop_f_clean[p_idx] - f0) * dt) * RAD2DEG
        if kin_jump_deg > max_kin_jump: max_kin_jump = kin_jump_deg

    for k in range(n_i):
        owner = i_owner_pmu[k]
        bi, bj = bi_arr[k], bj_arr[k]
        vi_prev = prop_v_clean[bi]
        vj_prev = prop_v_clean[bj]
        ys_c = ys_re[k] + 1j * ys_im[k]
        ym_c = ym_re[k] + 1j * ym_im[k]
        i_prev = ys_c * vi_prev + ym_c * vj_prev

        v_par = z_t[i_parent_v_idx[k]]
        i_br = z_t[i_meas_idx[k]]
        ym_mag = np.hypot(ym_re[k], ym_im[k])
        du_km = np.abs((i_br / v_par) - (i_prev / vi_prev)) / (ym_mag + 1e-9)

        if (du_km > 0.035) or (ws_r_adc_fdia[owner] > 0.05):
            if du_km > 0.035:
                r_br = du_km * 10.0
                if r_br > ws_r_adc_fdia[owner]:
                    ws_r_adc_fdia[owner] = r_br
                ws_w_vm_eff[owner] = w_vm_base[owner] * 0.15
            ws_w_pq_eff[k] = w_pq_base[k] * 0.05
            vi_clean = vi_prev * rot_coi
            vj_clean = vj_prev * rot_coi
            i_clean = ys_c * vi_clean + ym_c * vj_clean
            s_clean = vi_clean * np.conj(i_clean)
            ws_p_target[k], ws_q_target[k] = s_clean.real, s_clean.imag
        else:
            ws_w_pq_eff[k] = w_pq_base[k]
            s_br = v_par * np.conj(i_br)
            ws_p_target[k], ws_q_target[k] = s_br.real, s_br.imag

    for p_idx in range(n_pmu):
        if ws_r_adc_fdia[p_idx] > max_r_adc:
            max_r_adc = ws_r_adc_fdia[p_idx]
        if ws_r_adc_fdia[p_idx] > 0.05:
            ws_vm_target[p_idx] = np.abs(prop_v_clean[pmu_buses[p_idx]])
        else:
            ws_vm_target[p_idx] = ws_vm_pmu_raw[p_idx]

    if (pa_deg < pa_threshold_deg) and (max_r_adc < 0.05) and (max_kin_jump < 0.35):
        for k in range(nb):
            prop_v_clean[k] = ws_v_lse[k]
            v_aligned_out[k] = ws_v_lse[k]
        for p_idx in range(n_pmu):
            ang_clean = np.angle(ws_v_lse[pmu_buses[p_idx]])
            prop_va_pmu_clean[p_idx] = ang_clean
            prop_anchor_va_pmu[p_idx] = ang_clean
            prop_f_clean[p_idx] = 0.8 * prop_f_clean[p_idx] + 0.2 * f_meas_t[p_idx]
            est_spoof_deg_out[p_idx] = 0.0
        prop_anchor_state[0] = np.angle(ws_v_lse[slack_idx])
        prop_anchor_state[1] = 0.0
        return

    anchor_coi_int = prop_anchor_state[1] + d_ang_coi
    prop_anchor_state[1] = anchor_coi_int

    jit_solve_scalar_nls_sparse(
        ws_vm_target, ws_p_target, ws_q_target, ws_w_vm_eff, ws_w_pq_eff,
        pmu_buses, bi_arr, bj_arr, ys_re, ys_im, ym_re, ym_im, prop_v_clean, nb, True, 4, inv_perm,
        ws_x_curr, ws_w_irls_sq, ws_va, ws_vm, ws_v_re, ws_v_im, ws_JTJ, ws_JTr, ws_dx,
        ws_col_nz_idx, ws_col_nz_cnt, ws_idx_u, ws_jp_u, ws_jq_u, ws_v_rob
    )

    ang_clean_slack = np.angle(prop_v_clean[slack_idx])
    ang_rob_slack = np.angle(ws_v_rob[slack_idx])
    ang_anchor_slack = prop_anchor_state[0]

    min_mahal_sq = 1e30
    for p_idx in range(n_pmu):
        b_idx = pmu_buses[p_idx]
        v_rob_p = ws_v_rob[b_idx]
        ang_rob_p, vm_rob_p = np.angle(v_rob_p), np.abs(v_rob_p)
        ang_rob_rel = wrap_rad(ang_rob_p - ang_rob_slack)

        ang_prev_rel = wrap_rad(prop_va_pmu_clean[p_idx] - ang_clean_slack)
        d_theta_phys_rel = wrap_rad(ang_rob_rel - ang_prev_rel)
        ang_pred_step = prop_va_pmu_clean[p_idx] + d_theta_phys_rel + d_ang_coi

        ang_anc_rel = wrap_rad(prop_anchor_va_pmu[p_idx] - ang_anchor_slack)
        d_theta_phys_anc = wrap_rad(ang_rob_rel - ang_anc_rel)
        ang_pred_anc = prop_anchor_va_pmu[p_idx] + d_theta_phys_anc + anchor_coi_int

        ang_raw = ws_ang_pmu_raw[p_idx]
        ws_delta_diff_deg[p_idx] = wrap_rad(ang_raw - ang_rob_p) * RAD2DEG

        r_kin_step = abs(wrap_rad(ang_raw - ang_pred_step)) * RAD2DEG
        r_kin_anc = abs(wrap_rad(ang_raw - ang_pred_anc)) * RAD2DEG
        r_kin_deg = r_kin_step if r_kin_step >= r_kin_anc else r_kin_anc

        r_mag_v = abs(ws_vm_pmu_raw[p_idx] - vm_rob_p)
        f_phys_p = f_coi_est + d_theta_phys_rel * inv_two_pi_dt
        rocof_phys_p = (f_phys_p - prop_f_clean[p_idx]) / dt
        r_swing = abs(f_meas_t[p_idx] - f_phys_p) + 0.05 * abs(rocof_meas_t[p_idx] - rocof_phys_p)
        r_adc, r_dsp = ws_r_adc_fdia[p_idx], ws_r_dsp_f[p_idx]

        q1, q2, q3, q4, q5 = r_kin_deg / 0.22, r_adc / 0.04, r_mag_v / 0.006, r_dsp / 0.08, r_swing / 0.04
        mahal_sq = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4 + q5 * q5
        ws_sub_vals[p_idx] = mahal_sq
        if mahal_sq < min_mahal_sq:
            min_mahal_sq = mahal_sq

    for p_idx in range(n_pmu):
        mahal_sq = ws_sub_vals[p_idx]
        ws_trust_weights[p_idx] = np.exp(-0.5 * (mahal_sq - min_mahal_sq)) + 1e-6 / (1.0 + mahal_sq)

    n_clusters = fast_1d_sort_cluster_inplace(ws_delta_diff_deg, 0.12, ws_labels, ws_order)
    for c_id in range(n_clusters): ws_cluster_trust[c_id] = 0.0
    for p_idx in range(n_pmu): ws_cluster_trust[ws_labels[p_idx]] += ws_trust_weights[p_idx]

    best_cluster, best_trust = 0, ws_cluster_trust[0]
    for c_id in range(1, n_clusters):
        if ws_cluster_trust[c_id] > best_trust:
            best_trust, best_cluster = ws_cluster_trust[c_id], c_id

    sub_cnt = 0
    for p_idx in range(n_pmu):
        if (ws_labels[p_idx] == best_cluster) and (ws_r_adc_fdia[p_idx] < 0.05):
            ws_sub_vals[sub_cnt] = ws_delta_diff_deg[p_idx]
            ws_sub_weights[sub_cnt] = ws_trust_weights[p_idx]
            sub_cnt += 1

    if sub_cnt > 0:
        delta_corr_deg = weighted_median_1d_subset(ws_sub_vals, ws_sub_weights, ws_order, sub_cnt)
    else:
        delta_corr_deg = weighted_median_1d_subset(ws_delta_diff_deg, ws_trust_weights, ws_order, n_pmu)

    delta_corr_rad = delta_corr_deg * DEG2RAD
    rot_align = np.cos(delta_corr_rad) + 1j * np.sin(delta_corr_rad)
    for k in range(nb):
        val = ws_v_rob[k] * rot_align
        v_aligned_out[k], prop_v_clean[k] = val, val

    trusted_cnt = 0
    for p_idx in range(n_pmu):
        prop_va_pmu_clean[p_idx] = np.angle(v_aligned_out[pmu_buses[p_idx]])
        if (ws_labels[p_idx] == best_cluster) or (ws_r_adc_fdia[p_idx] >= 0.05):
            est_spoof_deg_out[p_idx] = 0.0
        else:
            est_spoof_deg_out[p_idx] = wrap_deg(ws_delta_diff_deg[p_idx] - delta_corr_deg)

        if ws_trust_weights[p_idx] > 0.3:
            f_upd = 0.7 * prop_f_clean[p_idx] + 0.3 * f_meas_t[p_idx]
            prop_f_clean[p_idx] = f_upd
            ws_sub_vals[trusted_cnt] = f_upd
            trusted_cnt += 1

    if trusted_cnt > 0:
        med_trusted_f = jit_median_1d(ws_sub_vals, ws_med_buf, trusted_cnt)
        for p_idx in range(n_pmu):
            if ws_trust_weights[p_idx] <= 0.3: prop_f_clean[p_idx] = med_trusted_f
