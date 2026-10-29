# Provably Resilient Synchrophasor State Estimation

**Authors:** Mohd Aarish Shaheen (UoK / IIT Bombay) and S. A. Soman (IIT Bombay)

This repository contains the official Python simulation pipeline, Numba JIT-compiled estimators, and LaTeX source files for the paper: *"Provably Resilient Synchrophasor State Estimation Under Coordinated Majority GPS Spoofing, Load Transients, and Substation Cyber-Physical Attacks."*

## Project Architecture

This project is built around a fully automated, reproducible pipeline. The Python engine executes the attack scenarios, exports real-time performance metrics directly into LaTeX macros, generates vector figures, and compiles the final academic manuscript.

* **`src/`** — Core simulation and plotting modules:
* `grid_models.py`: Canonical MATPOWER IEEE 30-bus and 118-bus grid topologies via `pandapower`.
* `run_simulation.py`: The live Numba simulation engine. Generates the 250-snapshot tape, runs the 6D Robust DSE vs. control algorithms, and exports both raw `.npz` arrays and `macros_metrics.tex`.
* `plot_figures.py`: Consumes the `.npz` output to generate dynamic `.svg` and `.pdf` vector graphics.


* **`tex/`** — Contains the LaTeX source (`main.tex`, `references.bib`, `IEEEtran.cls`). The manuscript dynamically integrates the Python-generated macros, ensuring all inline claims and tables are tied directly to the latest simulation run.
* **`data/`** — Stores the raw simulation output (`benchmark_results.npz`).

## Dependencies

The simulation framework requires the following Python packages:

* `numpy`, `scipy`, `matplotlib`
* `numba` (for JIT execution speed)
* `pandapower` (for grid modeling)

To compile the manuscript locally, a standard LaTeX distribution (e.g., TeX Live) featuring `latexmk` and `texlive-publishers` is required.

## Execution / Build Instructions

Run the end-to-end pipeline from the repository root to execute the math, update the figures, and compile the PDF manuscript:

```bash
# 1. Run the simulation and generate LaTeX macros
env PYTHONPATH=. python3 src/run_simulation.py

# 2. Generate updated vector figures
env PYTHONPATH=. python3 src/plot_figures.py

# 3. Compile the manuscript
cd tex
latexmk -C  # Clear old cache
latexmk -pdf -interaction=nonstopmode main.tex

```
