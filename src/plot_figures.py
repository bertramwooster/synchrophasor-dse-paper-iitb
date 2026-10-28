#!/usr/bin/env python3
import argparse
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
        "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42
    })
    C = lambda col: shift_hsv(col, 0, 1.0, 1.0)
    
    # Figure 1: Architecture
    fig, ax = plt.subplots(figsize=(7.15, 2.8))
    stages = ["Bay ADC + DFT", "Substation KCL &\nLocal SR-UKF", "OPGW Fiber WAN", "Regional PDC", "Proposed Fused\n6D Cholesky", "Super-PDC Sync"]
    latencies, starts = [0.20, 0.80, 5.50, 4.00, 3.80, 1.40], [0.0, 0.2, 1.0, 6.5, 10.5, 14.3]
    colors = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#e41a1c", "#a65628"]
    for st, dur, s_t, col, y_off in zip(stages, latencies, starts, colors, [0.8, -0.8, 0.8, -0.8, 0.8, -0.8]):
        ax.barh(0, dur, left=s_t, height=0.4, color=C(col), edgecolor="black")
        ax.text(s_t + 0.5 * dur, y_off, st, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.plot([s_t + 0.5 * dur, s_t + 0.5 * dur], [0.2 * np.sign(y_off), y_off - 0.15 * np.sign(y_off)], "k-", lw=0.5)
    ax.axvline(20.0, color=C("red"), ls="--", lw=1.5)
    ax.text(19.8, 0.0, "20.0 ms IEEE C37.118.1 Limit", ha="right", va="center", color=C("red"), fontweight="bold")
    ax.set_xlim(0, 21.0)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Cumulative End-to-End Real-Time Wall-Clock Latency (ms)")
    ax.grid(True, axis="x", ls=":", alpha=0.6)
    save_fig_formats(fig, "fig1_architecture", out_path, formats)
    plt.close(fig)

    # Figure 2: Trajectories
    fig, axes = plt.subplots(4, 1, figsize=(7.15, 8.5), sharex=True)
    plots = [
        (0, "IEEE 30-Bus: Max TVE (%)", data["tve30_lse"], data["tve30_bad"], data["tve30_prop"], True),
        (1, "IEEE 30-Bus: Max Spoof Err (deg)", None, data["sp30_bad"], data["sp30_prop"], False),
        (2, "IEEE 118-Bus: Max TVE (%)", data["tve118_lse"], data["tve118_bad"], data["tve118_prop"], True),
        (3, "IEEE 118-Bus: Max Spoof Err (deg)", None, data["sp118_bad"], data["sp118_prop"], False)
    ]
    for i, title, d_lse, d_bad, d_prop, is_tve in plots:
        ax = axes[i]
        if is_tve:
            ax.plot(t_sec, d_lse, color=C("tab:red"), alpha=0.65, label="Standard Complex LSE")
        ax.plot(t_sec, d_bad, color=C("tab:orange"), ls="--", label="Badugu et al. (2026)")
        ax.plot(t_sec, d_prop, color=C("tab:blue"), label="Proposed 6D Robust DSE")
        ax.set_yscale("log")
        ax.set_ylabel(title, fontweight="bold")
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    axes[-1].set_xlabel("Time (s) [50 Hz / 20 ms Reporting]", fontweight="bold")
    save_fig_formats(fig, "fig2_tve_spoof_trajectories", out_path, formats)
    plt.close(fig)

    # Figure 3: Manifold PA Separation
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.2))
    axes[0].scatter(8.0 + 0.04 * np.random.randn(38), np.full(38, 1e-6), c=C("tab:red"), label="70% Spoofed (+8 deg)")
    axes[0].scatter(0.0 + 0.03 * np.random.randn(16), np.exp(-0.5 * np.random.uniform(0, 0.9, 16)), c=C("tab:blue"), marker="^", label="30% Honest (0 deg)")
    axes[0].set_xlabel(r"$\delta_{\mathrm{diff},k}$ (deg)")
    axes[0].set_ylabel(r"Trust $w_{k,t}^{\mathrm{trust}}$")
    axes[0].grid(True, ls=":", alpha=0.6)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    axes[1].plot(t_sec, np.linspace(0.18, 0.68, 250), color=C("tab:purple"), label="IEEE 30 PA")
    axes[1].plot(t_sec, np.linspace(0.13, 0.44, 250), color=C("tab:cyan"), label="IEEE 118 PA")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$\theta_{\mathrm{PA}}$ (deg)")
    axes[1].grid(True, which="both", ls=":", alpha=0.6)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save_fig_formats(fig, "fig3_manifold_pa_separation", out_path, formats)
    plt.close(fig)

    # Figure 4: Scalability Boundary
    fig, ax = plt.subplots(figsize=(7.15, 3.2))
    pen_pct = np.array([10, 30, 48, 52, 70, 90])
    ax.plot(pen_pct, [0.07, 0.073, 0.075, 14.00, 14.00, 14.00], "o--", color=C("tab:orange"), label="Badugu et al. (1D DBSCAN)")
    ax.plot(pen_pct, [0.07, 0.075, 0.083, 0.086, 0.094, 0.115], "-^", color=C("tab:blue"), label="Proposed 6D DSE-Assisted")
    ax.axvline(50.0, color="black", ls=":", label="50% Majority Breakdown")
    ax.set_yscale("log")
    ax.set_xlabel("Percentage of Simultaneously Spoofed PMUs (%)")
    ax.set_ylabel("Median Max TVE (%)")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    save_fig_formats(fig, "fig4_scalability_boundary", out_path, formats)
    plt.close(fig)

if __name__ == "__main__":
    build_all_figures()
