import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
from dataLoader import load_all_txt_files  
import os


DATA_DIR = './dataSets'          
RESULTS_DIR = './results'    
os.makedirs(RESULTS_DIR, exist_ok=True)

R = 8.314  
alpha_levels = np.linspace(0.1, 0.9, 9)


print(f" Loading .txt files from: {DATA_DIR}")
data_list = load_all_txt_files(DATA_DIR)

if not data_list:
    print(" No valid data found in directory. Please check your TXT files.")
    exit()


data = {}

for filename, temp, weight in data_list:
    
    import re
    match = re.search(r'@(\d+)', filename)
    if not match:
        print(f" Could not extract heating rate from {filename}, skipping.")
        continue
    beta = int(match.group(1))
    T = temp + 273.15  
    w = np.array(weight)
    alpha = (w[0] - w) / (w[0] - w[-1])
    data[beta] = {"T": T, "alpha": alpha}


plt.figure(figsize=(8, 6))
summary = []

for a in alpha_levels:
    X, Y = [], []
    for beta, rec in data.items():
        fT = interp1d(rec["alpha"], rec["T"], bounds_error=False, fill_value=np.nan)
        T_a = fT(a)
        if np.isfinite(T_a):
            X.append(1.0 / T_a)
            Y.append(np.log(beta / (T_a**2)))

    if len(X) >= 2:
        X, Y = np.array(X), np.array(Y)
        slope, intercept, r, *_ = linregress(X, Y)
        Ea = -slope * R  

        summary.append({
            "α": round(a, 2),
            "slope": slope,
            "intercept": intercept,
            "R²": r**2,
            "Ea (kJ/mol)": Ea / 1000
        })

        plt.plot(X, Y, 'o', label=f'α={a:.2f}')
        plt.plot(X, slope * X + intercept, '--')


plt.xlabel("1/T (1/K)")
plt.ylabel("ln(β / T²)")
plt.title("KAS Analysis — ln(β/T²) vs 1/T for Different α")
plt.legend(fontsize='small', ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()


df = pd.DataFrame(summary)
csv_path = os.path.join(RESULTS_DIR, "kas_analysis.csv")
df.to_csv(csv_path, index=False)

print("\n Activation Energy Summary:")
print(df)
print(f"\n Results saved to {csv_path}")
