import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
from tqdm import tqdm
import os, re
from dataLoader import load_all_txt_files


R = 8.314  
alpha_levels = np.linspace(0.2, 0.8, 7)
DATA_DIR = '../dataSets'
RESULTS_DIR = '../results'
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
        print(f" Skipping {fname} — insufficient points")
        continue

    T_K = Ts + 273.15

    time = np.arange(len(Ts))

    w0, wf = weight[0], weight[-1]
    alpha = (w0 - weight) / (w0 - wf)

    dadt = np.gradient(alpha, time)

    data[beta] = {"alpha": alpha, "dadt": dadt, "T": T_K}


# === PRECOMPUTE INTERPOLATORS (for speed) ===
interpolators = {}
for beta, rec in data.items():
    alpha = rec["alpha"]
    T = rec["T"]
    dadt = rec["dadt"]

    step = max(1, len(alpha) // 500)
    alpha_ds = alpha[::step]
    T_ds = T[::step]
    dadt_ds = dadt[::step]

    fT = interp1d(alpha_ds, T_ds, fill_value="extrapolate", assume_sorted=True)
    fd = interp1d(alpha_ds, dadt_ds, fill_value="extrapolate", assume_sorted=True)
    interpolators[beta] = (fT, fd)


kB = 1.381e-23  # J/K
h = 6.626e-34   # J·s

records = []
plt.figure(figsize=(9, 7))

for a in tqdm(alpha_levels, desc="Processing α levels"):
    X, Y, T_list = [], [], []

    for beta, (fT, fd) in interpolators.items():
        try:
            T_a = float(fT(a))
            da_a = float(fd(a))
        except Exception:
            continue

        if np.isfinite(T_a) and np.isfinite(da_a) and da_a > 0:
            X.append(1.0 / T_a)
            Y.append(np.log(beta * da_a))
            T_list.append(T_a)

    if len(X) >= 2:
        X = np.array(X)
        Y = np.array(Y)
        slope, intercept, r, *_ = linregress(X, Y)
        Ea_J = -slope * R   # J/mol
        Ea_kJ = Ea_J / 1000

        Tm = np.mean(T_list)
        beta_avg = np.mean(list(data.keys()))

        k0 = (beta_avg * Ea_J * np.exp(Ea_J / (R * Tm))) / (R * (Tm ** 2))

        delta_H = (Ea_J - R * Tm) / 1000  # kJ/mol
        delta_G = (Ea_J + R * Tm * np.log((kB * Tm) / (h * k0))) / 1000  # kJ/mol
        delta_S = (delta_H * 1000 - delta_G * 1000) / Tm / 1000  # kJ/mol·K

        records.append({
            "α": round(a, 2),
            "Ea (kJ/mol)": round(Ea_kJ, 3),
            "Intercept": intercept,
            "R²": round(r ** 2, 5),
            "k0 (1/min)": f"{k0:.3e}",
            "ΔH (kJ/mol)": round(delta_H, 3),
            "ΔG (kJ/mol)": round(delta_G, 3),
            "ΔS (kJ/mol·K)": round(delta_S, 5)
        })

        label = f'α={a:.2f}'
        plt.plot(X, Y, 'o', label=label)
        plt.plot(X, slope * X + intercept, '--')


df_res = pd.DataFrame(records)
csv_path = os.path.join(RESULTS_DIR, "dfm_results.csv")
df_res.to_csv(csv_path, index=False)

plt.xlabel("1/T (1/K)", fontsize=12)
plt.ylabel("ln[β · dα/dt]", fontsize=12)
plt.title("Friedman DFM Analysis", fontsize=14)
plt.legend(fontsize='x-small', ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

print("\n Friedman DFM Summary :")
print(df_res.head().to_string(index=False))
print(f" Results saved to: {csv_path}")

