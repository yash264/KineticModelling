import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from dataLoader import load_all_txt_files  
import os


DATA_DIR = './dataSets'  
heating_rates = [10, 20, 40, 60]  


def compute_alpha(weight):
    w0, wf = weight[0], weight[-1]
    return (w0 - weight) / (w0 - wf)


def daem_model(T, E0, sigma, A, beta):
    R = 8.314
    T_K = T + 273.15

    E_range = np.linspace(E0 - 3 * sigma, E0 + 3 * sigma, 100)
    f_E = np.exp(-(E_range - E0)**2 / (2 * sigma**2))
    f_E /= trapezoid(f_E, E_range)

    exp_term = np.exp(-A / beta * np.exp(-E_range[None, :] / (R * T_K[:, None])))
    alpha_T = 1 - np.trapz(f_E * exp_term, E_range, axis=1)
    return alpha_T


def fit_daem(temp, alpha, beta):
    p0 = [150e3, 20e3, 1e13]
    bounds = ([50e3, 5e3, 1e10], [300e3, 100e3, 1e16])

    def model_to_fit(T, E0, sigma, A):
        return daem_model(T, E0, sigma, A, beta)

    popt, _ = curve_fit(model_to_fit, temp, alpha, p0=p0, bounds=bounds, maxfev=2000)
    alpha_fit = daem_model(temp, *popt, beta)
    r2 = 1 - np.sum((alpha - alpha_fit)**2) / np.sum((alpha - np.mean(alpha))**2)
    return popt, alpha_fit, r2


print(f" Loading files from: {DATA_DIR}")
data_list = load_all_txt_files(DATA_DIR)

if not data_list:
    print("No valid data found. Please check your data directory.")
    exit()

results = []
plt.figure(figsize=(12, 8))

for i, (filename, temp, weight) in enumerate(data_list):
    alpha = compute_alpha(weight)
    beta = heating_rates[i % len(heating_rates)]  
    print(f"  Fitting {filename} at {beta}°C/min ...")

    popt, alpha_fit, r2 = fit_daem(temp, alpha, beta)
    E0, sigma, A = popt
    results.append({
        "File": filename,
        "β (°C/min)": beta,
        "E₀ (kJ/mol)": E0 / 1000,
        "σ (kJ/mol)": sigma / 1000,
        "A (1/s)": A,
        "R²": r2
    })

    plt.plot(temp, alpha, 'o', ms=3, label=f'{beta}°C/min data')
    #plt.plot(temp, alpha_fit, '-', lw=1.5, label=f'{beta}°C/min fit')


plt.xlabel("Sample Temperature (°C)", fontsize=12)
plt.ylabel("Conversion α", fontsize=12)
plt.title("DAEM Fit for Heating Rates at 600 μm", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


df_results = pd.DataFrame(results)
print("\n DAEM Fitting Results:")
print(df_results.to_string(index=False))

df_results.to_csv("./results/daem_results.csv", index=False)
print("\n Results saved to daem_results.csv")



