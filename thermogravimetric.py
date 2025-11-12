import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import os, re
from analysis.dataLoader import load_all_txt_files


DATA_DIR = './dataSets'
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

ZONES = {"Zone I": (0, 300), "Zone II": (300, 600), "Zone III": (600, 900)}


datasets = load_all_txt_files(DATA_DIR)
data = {}

for i, (fname, Ts, weight) in enumerate(datasets):
    m = re.search(r'@(\d+)', fname)
    beta = float(m.group(1)) if m else [10, 20, 35, 55][i % 4]
    Ts, weight = np.asarray(Ts, float), np.asarray(weight, float)
    if len(Ts) < 20:
        continue
    data[beta] = {"T": Ts, "weight": weight}


T_grid = np.linspace(20, 900, 600)  

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
summary_rows = []


for beta, rec in sorted(data.items()):
    T, w = rec["T"], rec["weight"]

    if w[-1] > w[0]:
        w = w[0] - (w - w[0])
    w_pct = 100 * (w / w[0])

    w_interp = np.interp(T_grid, T, w_pct)

    w_smooth = savgol_filter(w_interp, 15, 3)

    dwdT = -np.gradient(w_smooth, T_grid)

    axes[0].plot(T_grid, w_smooth, lw=1.6, label=f'{beta} °C/min')
    axes[1].plot(T_grid, dwdT, lw=1.6, label=f'{beta} °C/min')


    for zone, (Tmin, Tmax) in ZONES.items():
        mask = (T_grid >= Tmin) & (T_grid <= Tmax)

        if not np.any(mask):
            continue
        
        Ti, Tf = T_grid[mask][0], T_grid[mask][-1]
        idx_max = np.argmax(dwdT[mask])
        DTmax = T_grid[mask][idx_max]
        DRmax = dwdT[mask][idx_max]

        summary_rows.append({
            "Zone": zone,
            "β (K min⁻¹)": beta,
            "Ti (K)": round(Ti + 273, 1),
            "Tf (K)": round(Tf + 273, 1),
            "DTmax (K)": round(DTmax + 273, 1),
            "DRmax (% min⁻¹)": round(DRmax, 2)
        })

zone_colors = ['#f8e9d7', '#d9eef5', '#ebdcf8']
for ax in axes:
    for i, (z, (Tmin, Tmax)) in enumerate(ZONES.items()):
        ax.axvspan(Tmin, Tmax, color=zone_colors[i], alpha=0.25)
        ax.text((Tmin + Tmax) / 2, ax.get_ylim()[1] * 0.9, z,
                ha='center', fontsize=10, weight='bold')

axes[0].set_title("(a) Thermogravimetric Profiles")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Mass (wt%)")
axes[1].set_title("(b) Differential Thermogravimetric Profiles")
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("DTG (wt%/°C)")
for ax in axes:
    ax.grid(True)
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "thermogravimetric.png"), dpi=300)
plt.show()


table = summary_df.pivot_table(
    index="β (K min⁻¹)",
    columns="Zone",
    values=["Ti (K)", "Tf (K)", "DTmax (K)", "DRmax (% min⁻¹)"]
)

os.makedirs(RESULTS_DIR, exist_ok=True)
csv_path = os.path.join(RESULTS_DIR, "thermogravimetric.csv")

table.to_csv(csv_path)
print(f"\n Summary Table Saved To: {os.path.abspath(csv_path)}\n")

