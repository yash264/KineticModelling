import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
import os, re
from dataLoader import load_all_txt_files


R = 8.314 
alpha_levels = np.linspace(0.1, 0.9, 9)
DATA_DIR = './dataSets'
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

datasets = load_all_txt_files(DATA_DIR)

data = {}

for i, (fname, Ts, weight) in enumerate(datasets):
    
    m = re.search(r'@(\d+)', fname)
    if m:
        beta = float(m.group(1))
    else:
        beta = [10, 20, 40, 60][i % 4]  

    if len(Ts) < 5 or len(weight) < 5:
        print(f" Skipping {fname} (insufficient data)")
        continue

    Ts = np.array(Ts, dtype=float)
    weight = np.array(weight, dtype=float)
    T_K = Ts + 273.15

    time = np.arange(len(Ts))

    w0, wf = weight[0], weight[-1]
    alpha = (w0 - weight) / (w0 - wf)

    dadt = np.gradient(alpha, time)

    data[beta] = {"alpha": alpha, "dadt": dadt, "T": T_K}


records = []
plt.figure(figsize=(9, 7))

for a in alpha_levels:
    X, Y = [], []

    for beta, rec in data.items():
        alpha = np.array(rec["alpha"])
        T = np.array(rec["T"])
        dadt = np.array(rec["dadt"])

        if a < np.nanmin(alpha) or a > np.nanmax(alpha):
            continue

        try:
            fT = interp1d(alpha, T, fill_value="extrapolate")
            fd = interp1d(alpha, dadt, fill_value="extrapolate")
            T_a = float(fT(a))
            da_a = float(fd(a))
        except Exception:
            continue

        if np.isfinite(T_a) and np.isfinite(da_a) and da_a > 0:
            X.append(1.0 / T_a)
            Y.append(np.log(beta * da_a))

    if len(X) >= 2:
        X = np.array(X)
        Y = np.array(Y)
        slope, intercept, r, *_ = linregress(X, Y)
        Ea = -slope * R

        records.append({
            "α": a,
            "Ea (kJ/mol)": Ea / 1000,
            "Intercept": intercept,
            "R²": r**2
        })

        label = f'α={a:.2f}'
        plt.plot(X, Y, 'o', label=label)
        plt.plot(X, slope * X + intercept, '--')


plt.xlabel("1/T (1/K)", fontsize=12)
plt.ylabel("ln[β · dα/dt]", fontsize=12)
plt.title("Friedman DFM Analysis", fontsize=14)
plt.legend(fontsize='x-small', ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()
print(f" Plot saved to {plot_path}")


df_res = pd.DataFrame(records)
csv_path = os.path.join(RESULTS_DIR, "dfm_results.csv")
df_res.to_csv(csv_path, index=False)

print("\n Friedman Analysis Results:")
print(df_res.to_string(index=False))
print(f"\n Results saved to {csv_path}")

