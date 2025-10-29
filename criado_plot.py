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
        print(f"Skipping {fname} insufficient points.")
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
        "F1": 1 / (1 - alpha),
        "F2": 1 / ((1 - alpha) ** 2),
        "F3": 1 / ((1 - alpha) ** 3),
        "R2": 1 / (2 * (1 - alpha) ** 0.5),
        "R3": 1 / (3 * (1 - alpha) ** (2 / 3)),
        "D1": 2 * alpha,  
        "D2": -np.log(1 - alpha),
        "D3": (2 * (1 - (1 - alpha) ** (1 / 3))) / (3 * (1 - alpha) ** (2 / 3))
    }


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
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

    ax = axs[idx]

    ax.plot(alpha, dadt_norm, 'o-', color='dimgray', lw=1.0, ms=3, alpha=0.8, label='Exp')

    models = creado_models(alpha)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for (name, f), c in zip(models.items(), colors):
        f_norm = f / np.interp(0.5, alpha, f)
        ax.plot(alpha, f_norm, '-', lw=1.2, color=c, label=name)

    ax.set_title(f"({chr(97 + idx)}) β = {beta} K/min", fontsize=12, fontweight='bold')
    ax.set_xlabel("Conversion (α)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Z(α)/Z(0.5)", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 2.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='best', ncol=2)

plt.suptitle("Criado’s master plot", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(RESULTS_DIR, "criado_plot_.png"), dpi=400)
plt.show()

print(f"\n Criado’s master plot saved to: {RESULTS_DIR}")

