import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Carica dati
df = pd.read_csv("log_run_v3.txt", sep=",")

# Filtra dati validi (IMPORTANTE: aggiungo filtro error)
df = df[(df["time_ms"] > 0) & (df["error"] >= 0)]

# Raggruppa
grouped = df.groupby("threads")["time_ms"]

threads_list = []
means = []
ci_errors = []

for threads, values in grouped:
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # CI 95%
    t_crit = stats.t.ppf(0.975, df=n-1)
    margin = t_crit * std / np.sqrt(n)
    
    threads_list.append(threads)
    means.append(mean)
    ci_errors.append(margin)

# Ordina per threads
threads_list = np.array(threads_list)
means = np.array(means)
ci_errors = np.array(ci_errors)

order = np.argsort(threads_list)
threads_list = threads_list[order]
means = means[order]
ci_errors = ci_errors[order]

# =========================
# 🔹 GRAFICO TEMPI
# =========================
plt.figure()

plt.xticks(range(1, int(max(threads_list)) + 1))
plt.errorbar(threads_list, means, yerr=ci_errors, marker='o', capsize=5)

# Linea target 100 ms
plt.axhline(y=100, linestyle='--', label='Target 100 ms')

plt.xlabel("Numero di threads")
plt.ylabel("Tempo medio (ms)")
plt.title("Tempo medio vs threads con CI 95%")
plt.legend()
plt.grid()

plt.show()

# =========================
# 🔹 CALCOLO SPEEDUP + CI
# =========================
T1 = means[threads_list == 1][0]
delta_T1 = ci_errors[threads_list == 1][0]

speedup = T1 / means

# propagazione errore
speedup_errors = speedup * np.sqrt(
    (delta_T1 / T1)**2 +
    (ci_errors / means)**2
)
plt.figure()


plt.xticks(range(1, int(max(threads_list)) + 1))
plt.errorbar(threads_list, speedup, yerr=speedup_errors,
             marker='o', capsize=5, label="Speedup reale")

# linea ideale
plt.plot(threads_list, threads_list, linestyle='--', label="Speedup ideale")

plt.xlabel("Numero di threads")
plt.ylabel("Speedup")
plt.title("Speedup vs threads (con CI 95%)")

plt.xticks(range(1, int(max(threads_list)) + 1))

plt.legend()
plt.grid()

plt.show()

# =========================
# 🔹 STAMPA RISULTATI
# =========================
print("\n===== RISULTATI =====")
print(f"{'thr':>3} {'n':>3} {'mean(ms)':>12} {'CI_low':>12} {'CI_high':>12} {'speedup':>10} {'S_low':>10} {'S_high':>10}")

for i in range(len(threads_list)):
    thr = threads_list[i]
    mean = means[i]
    ci = ci_errors[i]
    s = speedup[i]
    s_err = speedup_errors[i]
    
    print(f"{thr:>3} {len(df[df['threads']==thr]):>3} "
          f"{mean:>12.2f} {mean-ci:>12.2f} {mean+ci:>12.2f} "
          f"{s:>10.2f} {s-s_err:>10.2f} {s+s_err:>10.2f}")