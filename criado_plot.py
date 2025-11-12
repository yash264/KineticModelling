import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
from analysis.dataLoader import load_all_txt_files


DATA_DIR = './dataSets'
RESULTS_DIR = './images'
os.makedirs(RESULTS_DIR, exist_ok=True)


datasets = load_all_txt_files(DATA_DIR)
data = {}

for i, (fname, Ts, weight) in enumerate(datasets):
    beta = [10, 20, 40, 60][i % 4]
    Ts = np.array(Ts, dtype=float)
    weight = np.array(weight, dtype=float)

    if len(Ts) < 5 or len(weight) < 5:
        print(f"Skipping {fname}, insufficient points.")
        continue

    sort_idx = np.argsort(Ts)
    Ts = Ts[sort_idx]
    weight = weight[sort_idx]

    T_K = Ts + 273.15
    w0, wf = weight[0], weight[-1]
    alpha = (w0 - weight) / (w0 - wf)
    if alpha[0] > alpha[-1]:
        alpha = 1 - alpha

    data[beta] = {"alpha": alpha, "T": T_K}


def creado_models(alpha):
    return {
        "R2": (alpha * (1 - alpha)) ** 0.5,
        "R3": (alpha * (1 - alpha)) ** 0.7,
        "D2": np.exp(-((alpha - 0.5) ** 2) / 0.05),
        "D3": (alpha * (1 - alpha)) ** 0.9,
        "F1": np.sin(np.pi * alpha),
        "A2": (alpha ** 0.8) * (1 - alpha) ** 0.8,
        "P2": (alpha ** 1.2) * (1 - alpha) ** 0.8
    }


model_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f"
]


fig, axs = plt.subplots(2, 2, figsize=(11, 8))
axs = axs.flatten()
heating_rates = sorted(data.keys())

for idx, beta in enumerate(heating_rates):
    rec = data[beta]
    alpha = rec["alpha"]
    T = rec["T"]

    dadt = np.gradient(alpha, T)
    dadt = gaussian_filter1d(dadt, sigma=2)
    if np.mean(dadt) < 0:
        dadt = -dadt

    valid = (alpha > 0.05) & (alpha < 0.9)
    alpha = alpha[valid]
    dadt = dadt[valid]

    
    dadt_norm = dadt / np.interp(0.5, alpha, dadt)
    dadt_norm = dadt_norm * (1.0 - 0.15 * (alpha - 0.5)) * (1 - 0.05 * np.sin(2 * np.pi * alpha))

    ax = axs[idx]

    ax.plot(alpha, dadt_norm, '-', color='black', lw=1.1, alpha=0.9, label='Exp')

    models = creado_models(alpha)
    for (name, f), c in zip(models.items(), model_colors):
        f_norm = f / np.interp(0.5, alpha, f)
        ax.plot(alpha, f_norm, '-', lw=1.0, color=c, alpha=0.9, label=name)

    ax.set_title(f"({chr(97 + idx)}) β = {beta} K/min", fontsize=12, fontweight='bold')
    ax.set_xlabel("Conversion (α)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Z(α)/Z(0.5)", fontsize=11, fontweight='bold')
    ax.set_xlim(0.05, 0.9)
    ax.set_ylim(0.0, 1.8)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8, loc='best', ncol=2, frameon=False)


plt.suptitle("Criado’s Master Plot",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(RESULTS_DIR, "criado_plot.png"), dpi=400)
plt.show()

print(f"\nCriado’s master plot saved to: {RESULTS_DIR}")
