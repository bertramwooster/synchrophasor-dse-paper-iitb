import sys
import subprocess
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# Auto-install pandapower to survive runtime restarts
try:
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.converter.pypower import to_ppc
    from pandapower.pypower.makeYbus import makeYbus
except ImportError:
    print("[SETUP] Installing pandapower for MATPOWER test cases...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandapower"])
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.converter.pypower import to_ppc
    from pandapower.pypower.makeYbus import makeYbus

class CanonicalIEEEGrid:
    def __init__(self, system_type: int = 30):
        self.sys_type = system_type
        self.f0 = 50.0
        self.dt = 0.02

        if system_type == 30:
            net = pn.case_ieee30()
            self.pmu_buses_1idx = [2, 3, 6, 10, 11, 12, 15, 20, 23, 25, 27, 28, 29]
            self.pa_threshold_deg = 0.1443
        elif system_type == 118:
            net = pn.case118()
            self.pmu_buses_1idx = [
                1, 3, 4, 5, 6, 8, 9, 11, 12, 15, 17, 19, 21, 23, 25, 26, 28, 30,
                34, 35, 37, 40, 43, 45, 46, 49, 52, 54, 56, 59, 62, 63, 65, 68,
                70, 71, 75, 76, 77, 78, 80, 83, 85, 86, 89, 90, 92, 94, 96, 100,
                105, 108, 110, 114
            ]
            self.pa_threshold_deg = 0.1107
        else:
            raise ValueError("system_type must be 30 or 118")

        pp.runpp(net, calculate_voltage_angles=True, tol_mva=1e-8)
        ppc = net["_ppc"] if "_ppc" in net and net["_ppc"] is not None else to_ppc(net, init="results")

        self.baseMVA = float(ppc["baseMVA"])
        bus_mat = ppc["bus"]
        branch_mat = ppc["branch"]
        gen_mat = ppc["gen"]

        if branch_mat.shape[1] < 32:
            branch_mat = np.pad(branch_mat, ((0, 0), (0, 32 - branch_mat.shape[1])), mode="constant", constant_values=0.0)

        self.nb = bus_mat.shape[0]
        self.nl = branch_mat.shape[0]
        self.pmu_buses = np.array([b - 1 for b in self.pmu_buses_1idx], dtype=np.int64)
        self.n_pmu = len(self.pmu_buses)

        Ybus_sp, Yf_sp, Yt_sp = makeYbus(self.baseMVA, bus_mat, branch_mat)
        self.Ybus, self.Yf, self.Yt = Ybus_sp.toarray(), Yf_sp.toarray(), Yt_sp.toarray()

        self.bus_type = np.real(bus_mat[:, 1]).astype(int)
        self.Pd0 = np.real(bus_mat[:, 2]) / self.baseMVA
        self.Qd0 = np.real(bus_mat[:, 3]) / self.baseMVA
        self.Vm0 = net.res_bus["vm_pu"].to_numpy(dtype=np.float64)
        self.Va0 = np.deg2rad(net.res_bus["va_degree"].to_numpy(dtype=np.float64))

        self.Va0 = self.Va0 - self.Va0[0]
        self.V0_complex = self.Vm0 * np.exp(1j * self.Va0)

        I0_inj = self.Ybus @ self.V0_complex
        S0_inj = self.V0_complex * np.conj(I0_inj)
        self.Pg0 = S0_inj.real + self.Pd0
        self.Qg0 = S0_inj.imag + self.Qd0

        self.gen_buses = np.unique(np.real(gen_mat[:, 0]).astype(int))
        rng_h = np.random.default_rng(system_type)
        self.gen_H = rng_h.uniform(3.0, 6.0, size=len(self.gen_buses))
        self.gen_D = rng_h.uniform(1.2, 2.0, size=len(self.gen_buses))

        self.fbus = np.real(branch_mat[:, 0]).astype(np.int64)
        self.tbus = np.real(branch_mat[:, 1]).astype(np.int64)

        self._build_pmu_measurement_model_and_verify_topology()

    def _build_pmu_measurement_model_and_verify_topology(self):
        h_rows, pmu_idx_l, m_type_l, fb_l, tb_l, ys_l, ym_l, sig_l = [], [], [], [], [], [], [], []

        for p_idx, b_i in enumerate(self.pmu_buses):
            sig_v, sig_i = 0.0010, 0.0020
            row_v = np.zeros(self.nb, dtype=complex)
            row_v[b_i] = 1.0 + 0.0j
            h_rows.append(row_v)
            pmu_idx_l.append(p_idx); m_type_l.append(0); fb_l.append(b_i)
            tb_l.append(-1); ys_l.append(0.0j); ym_l.append(0.0j); sig_l.append(sig_v)

            for e in range(self.nl):
                if self.fbus[e] == b_i:
                    b_j = self.tbus[e]
                    row_i = self.Yf[e, :].copy()
                    y_s, y_m = self.Yf[e, b_i], self.Yf[e, b_j]
                elif self.tbus[e] == b_i:
                    b_j = self.fbus[e]
                    row_i = self.Yt[e, :].copy()
                    y_s, y_m = self.Yt[e, b_i], self.Yt[e, b_j]
                else: continue
                h_rows.append(row_i)
                pmu_idx_l.append(p_idx); m_type_l.append(1); fb_l.append(b_i)
                tb_l.append(b_j); ys_l.append(y_s); ym_l.append(y_m); sig_l.append(sig_i)

        self.H = np.vstack(h_rows)
        self.m = self.H.shape[0]
        self.meas_pmu_idx = np.array(pmu_idx_l, dtype=np.int64)
        self.meas_type = np.array(m_type_l, dtype=np.int64)
        self.meas_fbus = np.array(fb_l, dtype=np.int64)
        self.meas_tbus = np.array(tb_l, dtype=np.int64)
        self.meas_y_self = np.array(ys_l, dtype=complex)
        self.meas_y_mut = np.array(ym_l, dtype=complex)
        self.meas_sigma = np.array(sig_l, dtype=np.float64)

        self.v_meas_idx = np.ascontiguousarray(np.where(self.meas_type == 0)[0], dtype=np.int64)
        self.i_meas_idx = np.ascontiguousarray(np.where(self.meas_type == 1)[0], dtype=np.int64)
        self.i_owner_pmu = np.ascontiguousarray(self.meas_pmu_idx[self.i_meas_idx], dtype=np.int64)
        self.i_parent_v_idx = np.ascontiguousarray(self.v_meas_idx[self.i_owner_pmu], dtype=np.int64)

        self.bi_arr = np.ascontiguousarray(self.meas_fbus[self.i_meas_idx], dtype=np.int64)
        self.bj_arr = np.ascontiguousarray(self.meas_tbus[self.i_meas_idx], dtype=np.int64)
        self.ys_re = np.ascontiguousarray(self.meas_y_self[self.i_meas_idx].real, dtype=np.float64)
        self.ys_im = np.ascontiguousarray(self.meas_y_self[self.i_meas_idx].imag, dtype=np.float64)
        self.ym_re = np.ascontiguousarray(self.meas_y_mut[self.i_meas_idx].real, dtype=np.float64)
        self.ym_im = np.ascontiguousarray(self.meas_y_mut[self.i_meas_idx].imag, dtype=np.float64)
        self.w_vm_base = np.ascontiguousarray(1.0 / self.meas_sigma[self.v_meas_idx], dtype=np.float64)
        self.w_pq_base = np.ascontiguousarray(1.0 / (self.meas_sigma[self.i_meas_idx] * 1.05), dtype=np.float64)

        self.W_inv_sqrt = np.ascontiguousarray(1.0 / self.meas_sigma, dtype=np.float64)
        W_diag = self.W_inv_sqrt ** 2
        self.H_weighted = self.H * self.W_inv_sqrt[:, None]
        _, R_dense = la.qr(self.H_weighted, mode="economic")

        HW_dense = self.H.conj().T * W_diag[None, :]
        HW_sp = sp.csr_matrix(np.where(np.abs(HW_dense) > 1e-12, HW_dense, 0.0))
        self.hw_indptr, self.hw_indices, self.hw_data = np.ascontiguousarray(HW_sp.indptr, dtype=np.int64), np.ascontiguousarray(HW_sp.indices, dtype=np.int64), np.ascontiguousarray(HW_sp.data, dtype=np.complex128)

        R_clean = np.where(np.abs(R_dense) > 1e-12, R_dense, 0.0)
        RH_clean = R_clean.conj().T
        self.r_diag = np.ascontiguousarray(np.diag(R_clean), dtype=np.complex128)
        self.rh_diag = np.ascontiguousarray(np.diag(RH_clean), dtype=np.complex128)

        R_strict_upper = sp.csr_matrix(np.triu(R_clean, k=1))
        self.r_indptr, self.r_indices, self.r_data = np.ascontiguousarray(R_strict_upper.indptr, dtype=np.int64), np.ascontiguousarray(R_strict_upper.indices, dtype=np.int64), np.ascontiguousarray(R_strict_upper.data, dtype=np.complex128)

        RH_strict_lower = sp.csr_matrix(np.tril(RH_clean, k=-1))
        self.rh_indptr, self.rh_indices, self.rh_data = np.ascontiguousarray(RH_strict_lower.indptr, dtype=np.int64), np.ascontiguousarray(RH_strict_lower.indices, dtype=np.int64), np.ascontiguousarray(RH_strict_lower.data, dtype=np.complex128)

    def solve_ac_power_flow(self, Pd: np.ndarray, Qd: np.ndarray, Pg: np.ndarray, v_init: np.ndarray, coi_angle_rad: float = 0.0) -> np.ndarray:
        V = v_init.copy()
        Vm, Va = np.abs(V), np.angle(V)
        slack_idx = int(np.where(self.bus_type == 3)[0][0])
        Va[slack_idx] = self.Va0[slack_idx] + coi_angle_rad
        pv_buses = np.where(self.bus_type == 2)[0]
        pq_buses = np.where(self.bus_type == 1)[0]
        non_slack = np.sort(np.concatenate([pv_buses, pq_buses]))
        n_ns, n_pq = len(non_slack), len(pq_buses)
        P_spec, Q_spec = Pg - Pd, self.Qg0 - Qd
        for _ in range(7):
            V_phasor = Vm * np.exp(1j * Va)
            I_inj = self.Ybus @ V_phasor
            S_calc = V_phasor * np.conj(I_inj)
            mismatch = np.concatenate([P_spec[non_slack] - S_calc[non_slack].real, Q_spec[pq_buses] - S_calc[pq_buses].imag])
            if np.max(np.abs(mismatch)) < 1e-7: break
            diagV, diagI, diagVnorm = np.diag(V_phasor), np.diag(I_inj), np.diag(V_phasor / Vm)
            dS_dVa = 1j * diagV @ (np.conj(diagI) - np.conj(self.Ybus) @ np.conj(diagV))
            dS_dVm = diagV @ np.conj(self.Ybus @ diagVnorm) + diagVnorm @ np.conj(diagI)
            J = np.block([[dS_dVa[np.ix_(non_slack, non_slack)].real, dS_dVm[np.ix_(non_slack, pq_buses)].real],
                          [dS_dVa[np.ix_(pq_buses, non_slack)].imag,  dS_dVm[np.ix_(pq_buses, pq_buses)].imag]])
            dx = la.solve(J, mismatch)
            Va[non_slack] += dx[:n_ns]
            Vm[pq_buses] += dx[n_ns:]
        return Vm * np.exp(1j * Va)
