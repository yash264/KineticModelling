import numpy as np
import matplotlib.pyplot as plt
import os, re
from analysis.dataLoader import load_all_txt_files

DATA_DIR = './dataSets'
RESULTS_DIR = './images'
os.makedirs(RESULTS_DIR, exist_ok=True)

datasets = load_all_txt_files(DATA_DIR)

data = {}
for i, (fname, Ts, weight) in enumerate(datasets):
    
    m = re.search(r'@(\d+)', fname)
    if m:
        beta = float(m.group(1))
    else:
        beta = [10, 20, 40, 60][i % 4]

    Ts = np.array(Ts, dtype=float)
    weight = np.array(weight, dtype=float)

    if len(Ts) < 5 or len(weight) < 5:
        print(f"Skipping {fname} (too few points)")
        continue

    data[beta] = {"T": Ts, "weight": weight}


fig, axes = plt.subplots(1, 2, figsize=(13, 5))


for beta, rec in sorted(data.items()):
    T = rec["T"]
    w = rec["weight"]
    
    if w[-1] > w[0]:  
        w = w[0] - (w - w[0])
    weight_pct = 100 * (w / w[0])
    axes[0].plot(T, weight_pct, lw=2, label=f'{beta} °C/min')

axes[0].set_xlabel("Temperature (°C)", fontsize=12)
axes[0].set_ylabel("Mass (wt%)", fontsize=12)
axes[0].set_title("(a) Thermogravimetric (TG) Profiles", fontsize=13)
axes[0].grid(True)
axes[0].legend()


for beta, rec in sorted(data.items()):
    T = rec["T"]
    w = rec["weight"]

    if w[-1] > w[0]:  
        w = w[0] - (w - w[0])

    weight_pct = 100 * (w / w[0])
    dwdT = -np.gradient(weight_pct, T) 
    axes[1].plot(T, dwdT, lw=2, label=f'{beta} °C/min')

axes[1].set_xlabel("Temperature (°C)", fontsize=12)
axes[1].set_ylabel("DTG (wt%/°C)", fontsize=12)
axes[1].set_title("(b) Differential Thermogravimetric (DTG) Profiles", fontsize=13)
axes[1].grid(True)
axes[1].legend()

plt.suptitle("Thermogravimetric profiles at different heating rates", fontsize=15, y=0.98)
plt.savefig(os.path.join(RESULTS_DIR, "thermogravimetric.png"), dpi=400)
plt.tight_layout()
plt.show()

