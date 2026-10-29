import numpy as np
from typing import Tuple
from sklearn.cluster import DBSCAN
from src.grid_models import CanonicalIEEEGrid
from src.math_utils import TWO_PI, RAD2DEG
from src.sparse_solvers import jit_compute_pa_and_lse_sparse, jit_solve_scalar_nls_sparse
from src.robust_estimator import jit_step_proposed_robust

class FastCanonicalEstimators:
    def __init__(self, sys_model: CanonicalIEEEGrid):
        self.sys = sys_model
        nb, n_pmu, n_i = sys_model.nb, sys_model.n_pmu, len(sys_model.i_meas_idx)
        n_meas, n_states = n_pmu + 2 * n_i, 2 * nb - 1

        self.badugu_cnt = 0
        self.badugu_x_clean = sys_model.V0_complex.copy()

        self.prop_init_flag = np.zeros(1, dtype=np.int64)
        self.prop_v_clean = sys_model.V0_complex.copy()
        self.prop_f_clean = np.full(n_pmu, sys_model.f0, dtype=np.float64)
        self.prop_va_pmu_clean = np.angle(self.prop_v_clean[sys_model.pmu_buses]).astype(np.float64)
        self.prop_anchor_va_pmu = np.angle(self.prop_v_clean[sys_model.pmu_buses]).astype(np.float64)
        self.prop_anchor_state = np.array([np.angle(self.prop_v_clean[sys_model.slack_idx]), 0.0], dtype=np.float64)

        self.ws_b, self.ws_u, self.ws_v_lse, self.ws_v_rob, self.ws_v_aligned = np.empty(nb, dtype=np.complex128), np.empty(nb, dtype=np.complex128), np.empty(nb, dtype=np.complex128), np.empty(nb, dtype=np.complex128), np.empty(nb, dtype=np.complex128)
        self.ws_est_spoof_deg, self.ws_ang_pmu_raw, self.ws_vm_pmu_raw, self.ws_r_adc_fdia, self.ws_r_dsp_f, self.ws_w_vm_eff, self.ws_vm_target, self.ws_delta_diff_deg, self.ws_trust_weights, self.ws_cluster_trust, self.ws_sub_vals, self.ws_sub_weights, self.ws_med_buf = [np.empty(n_pmu, dtype=np.float64) for _ in range(13)]
        self.ws_w_pq_eff, self.ws_p_target, self.ws_q_target = np.empty(n_i, dtype=np.float64), np.empty(n_i, dtype=np.float64), np.empty(n_i, dtype=np.float64)
        self.ws_labels, self.ws_order = np.empty(n_pmu, dtype=np.int64), np.empty(n_pmu, dtype=np.int64)

        self.ws_x_curr, self.ws_JTr, self.ws_dx = np.empty(n_states, dtype=np.float64), np.empty(n_states, dtype=np.float64), np.empty(n_states, dtype=np.float64)
        self.ws_w_irls_sq = np.empty(n_meas, dtype=np.float64)
        self.ws_va, self.ws_vm, self.ws_v_re, self.ws_v_im = np.empty(nb, dtype=np.float64), np.empty(nb, dtype=np.float64), np.empty(nb, dtype=np.float64), np.empty(nb, dtype=np.float64)
        self.ws_JTJ = np.zeros((n_states, n_states), dtype=np.float64)
        self.ws_col_nz_idx = sys_model.sym_nz_idx
        self.ws_col_nz_cnt = sys_model.sym_nz_cnt
        self.ws_idx_u, self.ws_jp_u, self.ws_jq_u = np.empty(4, dtype=np.int64), np.empty(4, dtype=np.float64), np.empty(4, dtype=np.float64)

    def compute_pa_and_lse(self, z: np.ndarray) -> Tuple[float, np.ndarray]:
        pa_deg = jit_compute_pa_and_lse_sparse(
            z, self.sys.W_inv_sqrt, self.sys.hw_indptr, self.sys.hw_indices, self.sys.hw_data,
            self.sys.rh_indptr, self.sys.rh_indices, self.sys.rh_data, self.sys.rh_diag,
            self.sys.r_indptr, self.sys.r_indices, self.sys.r_data, self.sys.r_diag,
            self.ws_b, self.ws_u, self.ws_v_lse
        )
        return float(pa_deg), self.ws_v_lse.copy()

    def call_jit_nls(self, z_in: np.ndarray, v_warm: np.ndarray, w_vm_eff: np.ndarray, w_pq_eff: np.ndarray, robust_huber: bool) -> np.ndarray:
        v_pmu, i_br, v_par = z_in[self.sys.v_meas_idx], z_in[self.sys.i_meas_idx], z_in[self.sys.i_parent_v_idx]
        s_br = v_par * np.conj(i_br)
        np.abs(v_pmu, out=self.ws_vm_target)
        self.ws_p_target[:], self.ws_q_target[:] = s_br.real, s_br.imag

        jit_solve_scalar_nls_sparse(
            self.ws_vm_target, self.ws_p_target, self.ws_q_target, w_vm_eff, w_pq_eff,
            self.sys.pmu_buses, self.sys.bi_arr, self.sys.bj_arr, self.sys.ys_re, self.sys.ys_im, self.sys.ym_re, self.sys.ym_im,
            v_warm, self.sys.nb, robust_huber, 4 if robust_huber else 3, self.sys.inv_perm,
            self.ws_x_curr, self.ws_w_irls_sq, self.ws_va, self.ws_vm, self.ws_v_re, self.ws_v_im, self.ws_JTJ, self.ws_JTr, self.ws_dx,
            self.ws_col_nz_idx, self.ws_col_nz_cnt, self.ws_idx_u, self.ws_jp_u, self.ws_jq_u, self.ws_v_rob
        )
        return self.ws_v_rob.copy()

    def step_badugu_2026(self, z_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pa_deg, _ = self.compute_pa_and_lse(z_t)
        if pa_deg < self.sys.pa_threshold_deg: self.badugu_cnt = max(self.badugu_cnt - 1, 0)
        else: self.badugu_cnt += 1
        if self.badugu_cnt < 4:
            if self.badugu_cnt == 0: self.badugu_x_clean[:] = self.ws_v_lse
            return self.ws_v_lse.copy(), np.zeros(self.sys.n_pmu)

        v_nls = self.call_jit_nls(z_t, self.badugu_x_clean, self.sys.w_vm_base, self.sys.w_pq_base, robust_huber=False)
        ang_pmu, ang_nls = np.angle(z_t[self.sys.v_meas_idx]), np.angle(v_nls[self.sys.pmu_buses])
        delta_diff_deg = (((ang_pmu - ang_nls + np.pi) % TWO_PI) - np.pi) * RAD2DEG

        labels = DBSCAN(eps=0.12, min_samples=1).fit(delta_diff_deg.reshape(-1, 1)).labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        dom_label = unique_labels[np.argmax(counts)]

        delta_corr_deg = float(np.median(delta_diff_deg[labels == dom_label]))
        v_aligned = v_nls * np.exp(1j * np.deg2rad(delta_corr_deg))
        est_spoof_deg = ((delta_diff_deg - delta_corr_deg + 180.0) % 360.0) - 180.0
        est_spoof_deg[labels == dom_label] = 0.0
        return v_aligned, est_spoof_deg

    def step_proposed_robust(self, z_t: np.ndarray, f_meas_t: np.ndarray, rocof_meas_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        jit_step_proposed_robust(
            z_t, f_meas_t, rocof_meas_t, self.prop_init_flag, self.prop_v_clean, self.prop_va_pmu_clean, self.prop_f_clean,
            self.prop_anchor_va_pmu, self.prop_anchor_state,
            self.sys.pmu_buses, self.sys.v_meas_idx, self.sys.i_meas_idx, self.sys.i_owner_pmu, self.sys.i_parent_v_idx,
            self.sys.bi_arr, self.sys.bj_arr, self.sys.ys_re, self.sys.ys_im, self.sys.ym_re, self.sys.ym_im,
            self.sys.w_vm_base, self.sys.w_pq_base, self.sys.W_inv_sqrt,
            self.sys.hw_indptr, self.sys.hw_indices, self.sys.hw_data, self.sys.rh_indptr, self.sys.rh_indices, self.sys.rh_data, self.sys.rh_diag,
            self.sys.r_indptr, self.sys.r_indices, self.sys.r_data, self.sys.r_diag, self.sys.pa_threshold_deg, self.sys.f0, self.sys.dt, self.sys.nb, self.sys.slack_idx, self.sys.inv_perm,
            self.ws_b, self.ws_u, self.ws_v_lse, self.ws_v_rob, self.ws_ang_pmu_raw, self.ws_vm_pmu_raw, self.ws_r_adc_fdia, self.ws_r_dsp_f,
            self.ws_w_vm_eff, self.ws_w_pq_eff, self.ws_vm_target, self.ws_p_target, self.ws_q_target, self.ws_delta_diff_deg, self.ws_trust_weights, self.ws_labels, self.ws_order,
            self.ws_cluster_trust, self.ws_sub_vals, self.ws_sub_weights, self.ws_med_buf, self.ws_x_curr, self.ws_w_irls_sq, self.ws_va, self.ws_vm, self.ws_v_re, self.ws_v_im,
            self.ws_JTJ, self.ws_JTr, self.ws_dx, self.ws_col_nz_idx, self.ws_col_nz_cnt, self.ws_idx_u, self.ws_jp_u, self.ws_jq_u,
            self.ws_v_aligned, self.ws_est_spoof_deg
        )
        return self.ws_v_aligned, self.ws_est_spoof_deg
