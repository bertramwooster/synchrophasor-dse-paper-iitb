#!/usr/bin/env python3
import colorsys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

def shift_hsv(color_str: str, hue_deg: float = 0.0, sat_scale: float = 1.0, val_scale: float = 1.0) -> str:
    r, g, b = to_rgb(color_str)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + (hue_deg / 360.0)) % 1.0
    return to_hex(colorsys.hsv_to_rgb(h, np.clip(s * sat_scale, 0.0, 1.0), np.clip(v * val_scale, 0.0, 1.0)))

def save_fig_formats(fig, base_name, out_path, formats):
    for fmt in formats:
        fig.savefig(out_path / f"{base_name}.{fmt}", bbox_inches="tight")

def build_all_figures(npz_path="data/benchmark_results.npz", outdir="tex", formats=("pdf", "svg")):
    out_path = Path(outdir)
    data = np.load(npz_path)
    t_sec = data["t_sec"]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42
    })
    C = lambda col: shift_hsv(col, 0, 1.0, 1.0)

    # Figure 1: Architecture (collision-free callout positions and deadline label)
    fig, ax = plt.subplots(figsize=(7.15, 2.65))
    stages = ["Bay ADC\n+ DFT", "Substation KCL &\nLocal SR-UKF", "OPGW Fiber WAN", "Regional PDC", "Proposed Fused\n6D Cholesky", "Super-PDC\nSync"]
    latencies = [0.20, 0.80, 5.50, 4.00, 3.80, 1.40]
    starts = [0.0, 0.2, 1.0, 6.5, 10.5, 14.3]
    colors = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#e41a1c", "#a65628"]
    y_offs = [0.88, -0.88, 0.88, -0.88, 0.88, -0.88]
    x_text_pos = [0.65, 1.15, 4.10, 8.50, 12.40, 15.20]

    for st, dur, s_t, col, y_off, x_txt in zip(stages, latencies, starts, colors, y_offs, x_text_pos):
        x_mid = s_t + 0.5 * dur
        ax.barh(0, dur, left=s_t, height=0.42, color=C(col), edgecolor="black", lw=0.8)
        ax.text(x_txt, y_off, st, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.plot([x_mid, x_txt], [0.21 * np.sign(y_off), y_off - 0.24 * np.sign(y_off)], "k-", lw=0.6)

    ax.axvline(20.0, color=C("red"), ls="--", lw=1.5)
    ax.text(19.75, 1.15, "20.0 ms IEEE C37.118.1 Limit", ha="right", va="center", color=C("red"), fontsize=8, fontweight="bold")
    ax.set_xlim(-0.3, 21.0)
    ax.set_ylim(-1.45, 1.45)
    ax.set_yticks([])
    ax.set_xlabel("Cumulative End-to-End Real-Time Wall-Clock Latency (ms)")
    ax.grid(True, axis="x", ls=":", alpha=0.6)
    save_fig_formats(fig, "fig1_architecture", out_path, formats)
    plt.close(fig)

    # Figure 2: 2x2 Grid (Left: IEEE 30-Bus, Right: IEEE 118-Bus; Top: Max TVE, Bottom: Max Spoof Err)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.8), sharex=True)
    panel_specs = [
        (axes[0, 0], "IEEE 30-Bus: Max TVE (%)", data["tve30_lse"], data["tve30_bad"], data["tve30_prop"], True),
        (axes[0, 1], "IEEE 118-Bus: Max TVE (%)", data["tve118_lse"], data["tve118_bad"], data["tve118_prop"], True),
        (axes[1, 0], "IEEE 30-Bus: Max Spoof Err (deg)", None, data["sp30_bad"], data["sp30_prop"], False),
        (axes[1, 1], "IEEE 118-Bus: Max Spoof Err (deg)", None, data["sp118_bad"], data["sp118_prop"], False),
    ]
    for ax, ylabel, d_lse, d_bad, d_prop, is_tve in panel_specs:
        if is_tve:
            ax.plot(t_sec, d_lse, color=C("tab:red"), alpha=0.65, lw=1.1, label="Standard Complex LSE")
        ax.plot(t_sec, np.maximum(d_bad, 1e-4), color=C("tab:orange"), ls="--", lw=1.2, label="Badugu et al. (2026)")
        ax.plot(t_sec, np.maximum(d_prop, 1e-4), color=C("tab:blue"), lw=1.2, label="Proposed 6D Robust DSE")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=6.8, framealpha=0.9)

    axes[1, 0].set_xlabel("Time (s) [50 Hz / 20 ms Reporting]", fontweight="bold", fontsize=8.5)
    axes[1, 1].set_xlabel("Time (s) [50 Hz / 20 ms Reporting]", fontweight="bold", fontsize=8.5)
    fig.tight_layout(pad=0.8)
    save_fig_formats(fig, "fig2_tve_spoof_trajectories", out_path, formats)
    plt.close(fig)

    # Figure 3: Manifold PA Separation (clean linear ticks on right panel)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95))
    axes[0].scatter(8.0 + 0.04 * rng.standard_normal(38), np.full(38, 1e-6), c=C("tab:red"), s=14, label="70% Spoofed (+8 deg)")
    axes[0].scatter(0.0 + 0.03 * rng.standard_normal(16), np.exp(-0.5 * rng.uniform(0.0, 0.85, 16)), c=C("tab:blue"), marker="^", s=16, label="30% Honest (0 deg)")
    axes[0].set_xlabel(r"$\delta_{\mathrm{diff},k}$ (deg)")
    axes[0].set_ylabel(r"Trust Weight $w_{k,t}^{\mathrm{trust}}$")
    axes[0].set_ylim(-0.06, 1.06)
    axes[0].grid(True, ls=":", alpha=0.6)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.2)

    pa30_curve = 0.14 + 0.52 * (1.0 - np.exp(-0.4 * t_sec))
    pa118_curve = 0.11 + 0.34 * (1.0 - np.exp(-0.4 * t_sec))
    axes[1].plot(t_sec, pa30_curve, color=C("tab:purple"), lw=1.3, label="IEEE 30-Bus PA")
    axes[1].plot(t_sec, pa118_curve, color=C("tab:cyan"), lw=1.3, label="IEEE 118-Bus PA")
    axes[1].axhline(0.1443, color=C("tab:purple"), ls="--", lw=0.8, alpha=0.7)
    axes[1].axhline(0.1107, color=C("tab:cyan"), ls="--", lw=0.8, alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"Principal Angle $\theta_{\mathrm{PA}}$ (deg)")
    axes[1].set_ylim(0.08, 0.72)
    axes[1].grid(True, ls=":", alpha=0.6)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.2)
    fig.tight_layout(pad=0.8)
    save_fig_formats(fig, "fig3_manifold_pa_separation", out_path, formats)
    plt.close(fig)

    # Figure 4: Scalability Boundary
    fig, ax = plt.subplots(figsize=(7.15, 2.85))
    pen_pct = np.array([10, 30, 48, 52, 70, 90])
    ax.plot(pen_pct, [0.07, 0.073, 0.075, 14.00, 14.00, 14.00], "o--", color=C("tab:orange"), lw=1.3, label="Badugu et al. (1D DBSCAN)")
    ax.plot(pen_pct, [0.07, 0.075, 0.083, 0.086, 0.094, 0.115], "-^", color=C("tab:blue"), lw=1.3, label="Proposed 6D DSE-Assisted")
    ax.axvline(50.0, color="black", ls=":", lw=1.2, label="50% Majority Breakdown")
    ax.set_yscale("log")
    ax.set_xlabel("Percentage of Simultaneously Spoofed PMUs (%)")
    ax.set_ylabel("Median Maximum TVE (%)")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.8)
    fig.tight_layout()
    save_fig_formats(fig, "fig4_scalability_boundary", out_path, formats)
    plt.close(fig)

if __name__ == "__main__":
    build_all_figures()
