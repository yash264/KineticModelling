import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.interpolate import interp1d
import os
from analysis.dataLoader import load_all_txt_files


R = 8.314      # J/mol·K
kB = 1.381e-23 # J/K
h = 6.626e-34  # J·s

alpha_levels = np.arange(0.2, 0.81, 0.1)  
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def smooth_data(data, window_size=7):

    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()


def calculate_conversion(weight):

    w0, wf = weight[0], weight[-1]

    if abs(w0 - wf) < 1e-10:
        return np.zeros_like(weight)

    alpha = (w0 - weight) / (w0 - wf)
    return np.clip(alpha, 0, 1)


def create_interpolators(datasets):
    interpolators = {}

    for i, data_tuple in enumerate(datasets):
        if len(data_tuple) == 3:
            fname, Ts, weight = data_tuple
            Time = None
        elif len(data_tuple) == 4:
            fname, Ts, weight, Time = data_tuple
        else:
            raise ValueError(f"Dataset {i} has unexpected number of elements: {len(data_tuple)}")
        
        beta = [10, 20, 40, 60][i % 4] 

        Ts = np.array(Ts, dtype=float)
        weight = np.array(weight, dtype=float)

        if Time is None:
            Time = np.linspace(0, len(Ts)-1, len(Ts))

        T_K = Ts + 273.15
        alpha = calculate_conversion(weight)
        step = max(1, len(alpha)//500)

        fT = interp1d(alpha[::step], T_K[::step], fill_value="extrapolate", assume_sorted=True)
        interpolators[beta] = {"fT": fT, "alpha": alpha, "T": T_K, "weight": weight, "Time": Time}
    return interpolators


def generic_method(alpha_levels, interpolators, formula_func):
    records = []

    for a in alpha_levels:
        X, Y, T_list = [], [], []

        for beta, rec in interpolators.items():
            T = float(rec["fT"](a))
            if np.isfinite(T):
                x_val, y_val = formula_func(beta, T)
                X.append(x_val)
                Y.append(y_val)
                T_list.append(T)

        if len(X) >= 2:
            X, Y = np.array(X), np.array(Y)
            slope, intercept, r_value, *_ = linregress(X, Y)
            Ea_J = -slope * R
            Ea_kJ = Ea_J / 1000
            Tm = np.mean(T_list)
            beta_avg = np.mean(list(interpolators.keys()))

            k0 = (beta_avg * Ea_J * np.exp(Ea_J / (R * Tm))) / (R * (Tm ** 2))
            delta_H = (Ea_J - R * Tm) / 1000
            delta_G = (Ea_J + R * Tm * np.log((kB * Tm) / (h * k0))) / 1000
            delta_S = (delta_H * 1000 - delta_G * 1000) / Tm / 1000

            records.append({
                "α": round(a, 2),
                "Slope": slope,
                "Intercept": intercept,
                "R²": round(r_value**2, 5),
                "Ea (kJ/mol)": round(Ea_kJ, 3),
                "k0 (1/min)": f"{k0:.3e}",
                "ΔH (kJ/mol)": round(delta_H, 3),
                "ΔG (kJ/mol)": round(delta_G, 3),
                "ΔS (kJ/mol·K)": round(delta_S, 5)
            })

    return pd.DataFrame(records)

def dfm_method(alpha_levels, interpolators):
    records = []

    for a in alpha_levels:
        X, Y, T_list = [], [], []

        for beta, rec in interpolators.items():
            alpha_arr = rec["alpha"]
            Time = rec["Time"]
            dalpha_dt = smooth_data(np.gradient(alpha_arr, Time*60))  
            f_rate = interp1d(alpha_arr, dalpha_dt, fill_value="extrapolate", assume_sorted=True)
            rate = float(f_rate(a))
            T = float(rec["fT"](a))

            if np.isfinite(T) and rate > 1e-10:
                X.append(1/T)
                Y.append(np.log(rate))
                T_list.append(T)

        if len(X) >= 2:
            X, Y = np.array(X), np.array(Y)
            slope, intercept, r_value, *_ = linregress(X, Y)
            Ea_J = -slope * R
            Ea_kJ = Ea_J / 1000
            Tm = np.mean(T_list)
            beta_avg = np.mean(list(interpolators.keys()))

            k0 = (beta_avg * Ea_J * np.exp(Ea_J / (R * Tm))) / (R * (Tm ** 2))
            delta_H = (Ea_J - R * Tm) / 1000
            delta_G = (Ea_J + R * Tm * np.log((kB * Tm) / (h * k0))) / 1000
            delta_S = (delta_H * 1000 - delta_G * 1000) / Tm / 1000

            records.append({
                "α": round(a, 2),
                "Slope": slope,
                "Intercept": intercept,
                "R²": round(r_value**2, 5),
                "Ea (kJ/mol)": round(Ea_kJ, 3),
                "k0 (1/min)": f"{k0:.3e}",
                "ΔH (kJ/mol)": round(delta_H, 3),
                "ΔG (kJ/mol)": round(delta_G, 3),
                "ΔS (kJ/mol·K)": round(delta_S, 5)
            })

    return pd.DataFrame(records)


def daem_formula(beta, T):     return 1/T, np.log(beta / T**2)
def kas_formula(beta, T):      return 1/T, np.log(beta / T**2)
def ofw_formula(beta, T):      return 1/T, np.log(beta)
def starink_formula(beta, T):  return 1/T, np.log(beta / T**1.92)


def main():
    datasets = load_all_txt_files('./dataSets')
    interpolators = create_interpolators(datasets)
    results = []

    df_daem = generic_method(alpha_levels, interpolators, daem_formula)
    df_daem["Model"] = "DAEM"
    results.append(df_daem)

    df_kas = generic_method(alpha_levels, interpolators, kas_formula)
    df_kas["Model"] = "KAS"
    results.append(df_kas)

    df_ofw = generic_method(alpha_levels, interpolators, ofw_formula)
    df_ofw["Model"] = "OFW"
    results.append(df_ofw)

    df_starink = generic_method(alpha_levels, interpolators, starink_formula)
    df_starink["Model"] = "Starink"
    results.append(df_starink)

    df_friedman = dfm_method(alpha_levels, interpolators)
    df_friedman["Model"] = "Friedman"
    results.append(df_friedman)

    final_df = pd.concat(results, ignore_index=True)
    csv_path = os.path.join(RESULTS_DIR, "kinetic_results.csv")
    final_df.to_csv(csv_path, index=False)

    print(f"\n Kinetic analysis complete. Results saved to: {csv_path}")
    print("\n Sample results:")
    print(final_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
