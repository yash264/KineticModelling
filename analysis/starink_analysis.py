import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
from tqdm import tqdm
import os
from dataLoader import load_all_txt_files

R = 8.314  
alpha_levels = np.linspace(0.1, 0.9, 9)
DATA_DIR = '../dataSets'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)


datasets = load_all_txt_files(DATA_DIR)
data = {}

for i, (fname, Ts, weight) in enumerate(datasets):
    
    beta = [10, 20, 40, 60][i % 4]

    Ts = np.array(Ts, dtype=float)
    weight = np.array(weight, dtype=float)

    if len(Ts) < 5 or len(weight) < 5:
        print(f"Skipping {fname} (insufficient data)")
        continue

    T_K = Ts + 273.15
    w0, wf = weight[0], weight[-1]
    alpha = (w0 - weight) / (w0 - wf)

    data[beta] = {"alpha": alpha, "T": T_K}

# === Precompute interpolators for speed ===
interpolators = {}
for beta, rec in data.items():
    alpha = rec["alpha"]
    T = rec["T"]
    step = max(1, len(alpha) // 500)
    interpolators[beta] = interp1d(alpha[::step], T[::step], fill_value="extrapolate", assume_sorted=True)


records = []
plt.figure(figsize=(9,7))

for a in tqdm(alpha_levels, desc="Processing α levels"):
    X, Y = [], []

    for beta, fT in interpolators.items():
        try:
            T_a = float(fT(a))
        except Exception:
            continue

        if np.isfinite(T_a):
            X.append(1.0 / T_a)
            Y.append(np.log(beta / (T_a**1.92)))  

    if len(X) >= 2:
        X = np.array(X)
        Y = np.array(Y)

        slope, intercept, r, *_ = linregress(X, Y)
        Ea = -slope * R / 1.0008 / 1000  

        records.append({
            "α": round(a,2),
            "Slope": slope,
            "Intercept": intercept,
            "R²": r**2,
            "Ea (kJ/mol)": round(Ea,3)
        })

        plt.plot(X, Y, 'o', label=f'α={a:.2f}')
        plt.plot(X, slope*X + intercept, '--')


df_res = pd.DataFrame(records)
csv_path = os.path.join(RESULTS_DIR, "starink_results.csv")
df_res.to_csv(csv_path, index=False)


plt.xlabel("1/T (1/K)", fontsize=12)
plt.ylabel("ln(β / T¹·⁹²)", fontsize=12)
plt.title("Starink Method Analysis", fontsize=14)
plt.legend(fontsize='x-small', ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Results saved to: {csv_path}")
print("\nStarink Summary:")
print(df_res.to_string(index=False))
