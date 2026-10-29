import numpy as np
from typing import Dict
from src.grid_models import CanonicalIEEEGrid
from src.math_utils import TWO_PI

def generate_dynamic_tape(sys_model: CanonicalIEEEGrid, n_steps: int = 250, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    nb, m, n_pmu, dt = sys_model.nb, sys_model.m, sys_model.n_pmu, sys_model.dt

    v_true_tape = np.zeros((n_steps, nb), dtype=np.complex128)
    z_attack_tape = np.zeros((n_steps, m), dtype=np.complex128)
    f_meas_tape = np.zeros((n_steps, n_pmu), dtype=np.float64)
    rocof_meas_tape = np.zeros((n_steps, n_pmu), dtype=np.float64)
    true_spoof_deg_tape = np.zeros((n_steps, n_pmu), dtype=np.float64)

    v_curr = sys_model.V0_complex.copy()
    coi_angle, f_coi = 0.0, sys_model.f0
    ou_noise = np.zeros(nb)
    H_total, D_total = np.sum(sys_model.gen_H), np.sum(sys_model.gen_D)
    Pm_total = np.sum(sys_model.Pg0)
    prev_va_pmu = np.angle(v_curr[sys_model.pmu_buses])
    prev_f_pmu = np.full(n_pmu, sys_model.f0)

    for t in range(n_steps):
        ou_noise = 0.92 * ou_noise + 0.003 * rng.standard_normal(nb)
        ramp_factor = 1.0 + 0.10 * (t / n_steps)
        step_factor = 0.18 if (100 <= t < 150) else 0.0

        Pd_t = sys_model.Pd0 * (ramp_factor + ou_noise)
        Qd_t = sys_model.Qd0 * (ramp_factor + ou_noise)
        if step_factor > 0.0:
            step_buses = [6, 14, 29] if nb == 30 else [14, 33, 53, 79, 89, 115]
            for b_s in step_buses:
                Pd_t[b_s] *= (1.0 + step_factor)
                Qd_t[b_s] *= (1.0 + step_factor)

        dP_load = np.sum(Pd_t - sys_model.Pd0)
        Pm_total = 0.94 * Pm_total + 0.06 * (np.sum(sys_model.Pg0) + dP_load * 1.02)
        Pg_t = sys_model.Pg0.copy()
        Pg_t[sys_model.gen_buses] += (Pm_total - np.sum(sys_model.Pg0)) * (sys_model.gen_H / H_total)

        df_dt = (sys_model.f0 / (2.0 * H_total)) * ((Pm_total - np.sum(sys_model.Pg0)) - dP_load - D_total * (f_coi - sys_model.f0) * 0.05)
        f_coi += df_dt * dt
        coi_angle += TWO_PI * (f_coi - sys_model.f0) * dt

        v_curr = sys_model.solve_ac_power_flow(Pd_t, Qd_t, Pg_t, v_curr, coi_angle)
        v_true_tape[t] = v_curr

        va_pmu = np.angle(v_curr[sys_model.pmu_buses])
        if t == 0:
            f_local, rocof_local = np.full(n_pmu, sys_model.f0), np.zeros(n_pmu)
        else:
            d_theta = (va_pmu - prev_va_pmu + np.pi) % TWO_PI - np.pi
            f_local = sys_model.f0 + d_theta / (TWO_PI * dt)
            rocof_local = (f_local - prev_f_pmu) / dt
        prev_va_pmu, prev_f_pmu = va_pmu.copy(), f_local.copy()

        z_exact = sys_model.H @ v_curr
        z_att = z_exact + (rng.normal(0.0, sys_model.meas_sigma / np.sqrt(2.0)) + 1j * rng.normal(0.0, sys_model.meas_sigma / np.sqrt(2.0)))
        spoof_deg = np.zeros(n_pmu)
        f_rep = f_local + rng.normal(0.0, 0.0015, size=n_pmu)
        rocof_rep = rocof_local + rng.normal(0.0, 0.03, size=n_pmu)

        if 50 <= t < 100:
            spoof_map = {1: 0.70, 6: -0.80, 9: -0.80, 12: 0.90}
            for p_idx, ang_deg in spoof_map.items():
                spoof_deg[p_idx] = ang_deg
                if t == 50: f_rep[p_idx] += np.deg2rad(ang_deg) / (TWO_PI * dt)

        elif 100 <= t < 150:
            n_maj = int(np.ceil(0.70 * n_pmu))
            for p_idx in range(n_maj):
                spoof_deg[p_idx] = 8.00
                if t == 100: f_rep[p_idx] += np.deg2rad(8.00) / (TWO_PI * dt)

        elif 150 <= t < 200:
            for p_idx in [2, 5, 9]: spoof_deg[p_idx] = 5.00
            for fdia_pmu in [0, 3]:
                v_idx = sys_model.v_meas_idx[fdia_pmu]
                z_att[v_idx] *= 1.18 * np.exp(1j * np.deg2rad(12.0))
                i_indices = np.where((sys_model.meas_pmu_idx == fdia_pmu) & (sys_model.meas_type == 1))[0]
                if len(i_indices) > 0: z_att[i_indices[0]] *= 0.65 * np.exp(-1j * np.deg2rad(18.0))

        elif t >= 200:
            n_maj = int(np.ceil(0.70 * n_pmu))
            drift_deg = 0.45 + 2.50 * ((t - 200) * dt)
            for p_idx in range(n_maj):
                spoof_deg[p_idx] = drift_deg
                f_rep[p_idx] += np.deg2rad(2.50) / TWO_PI
                rocof_rep[p_idx] += 0.12

        for p_idx in range(n_pmu):
            if abs(spoof_deg[p_idx]) > 1e-9:
                z_att[sys_model.meas_pmu_idx == p_idx] *= np.exp(1j * np.deg2rad(spoof_deg[p_idx]))

        z_attack_tape[t] = z_att
        f_meas_tape[t] = f_rep
        rocof_meas_tape[t] = rocof_rep
        true_spoof_deg_tape[t] = spoof_deg

    return {"v_true": v_true_tape, "z_attack": z_attack_tape, "f_meas": f_meas_tape, "rocof_meas": rocof_meas_tape, "true_spoof_deg": true_spoof_deg_tape}
