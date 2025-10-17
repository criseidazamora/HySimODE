import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==== Parámetros ====
raftot = 0.25
mkktot = 0.25
erktot = 0.25

# Volúmenes (L)
vol_syn = 0.2e-15  # 0.2 fL
vol_dend = 2.0e-15  # 2.0 fL
NA = 6.022e23

t_start = 0
stim_times = [100, 105, 110]  # ajustados


# Tiempo
t_start = 0
t_end = 460
dt = 0.1
time_points = np.arange(t_start, t_end, dt)

# Estímulo
stim_times = [100, 105, 110]
Cabas = 0.04
AMPTETCA = 1.4
AMPTETCAD = 0.65

# ==== Parámetros modelo ====
params = {
    "kfck2": 200.0, "tauck2": 1.0, "Kck2": 1.4,
    "kfpp": 2.0, "taupp": 2.0, "Kpp": 0.225,
    "kphos1": 0.45, "kdeph1": 0.006,
    "kphos4": 2.0, "kdeph4": 0.011,
    "kphos5": 0.011, "kdeph5": 0.04,
    "kphos7": 4.0, "kdeph7": 0.1,
    "kphos8": 0.015, "kdeph8": 0.02,
    "ktrans1": 1.2, "vtrbas1": 0.00005, "tauprp": 45.0,
    "vtrbaspkm": 0.0003, "taupkm": 50.0,
    "kpkmon": 0.5, "kpkmon2": 0.055, "Kpkm": 0.75,
    "kexpkm": 0.0025, "klkpkm": 0.012, "Vsd": 0.03,
    "kltp": 0.014, "tfsyn": 30.0, "vfbas": 0.01,
    "kltd": 0.03, "tnsyn": 600.0, "vdbas": 0.0033,
    "kfbasraf": 0.0001, "kbraf": 0.12,
    "kfmkk": 0.6, "kbmkk": 0.025, "Kmkk": 0.25,
    "kferk": 0.52, "kberk": 0.025, "Kmk": 0.25
}

def stimulus(t):
    casyn, cadend = Cabas, Cabas
    if any(abs(t - stim) < 0.05 for stim in stim_times):
        casyn, cadend = AMPTETCA, AMPTETCAD
    return casyn, cadend


# ==== Sistema ODE ====
def smolen_odes(t, y):

    #if int(t) % 10 == 0:  # Solo imprime cada 10 minutos para evitar exceso
     #   print(f"Simulando en t = {t:.1f} min")

    (CaMKII, Raf_s, MEK_s, MEKpp_s, ERK_s, ERKpp_s, PP, tagp1,
     tagd1, tagd2, CaMK_d, Raf_d, MEK_d, MEKpp_d, ERK_d, ERKpp_d,
     psit1, psit2, PRP, PKM_d, PKM_s, fsyn, nsyn) = y

    p = params
    casyn, cadend = stimulus(t)

    powca = casyn**4
    powcad = cadend**4
    powkc = p["Kck2"]**4
    powkc2 = p["Kpp"]**4
    powkcd = 0.6**4

    # Totales
    Rafp_s = raftot - Raf_s
    MEKp_s = mkktot - MEK_s - MEKpp_s
    ERKp_s = erktot - ERK_s - ERKpp_s
    Rafp_d = raftot - Raf_d
    MEKp_d = mkktot - MEK_d - MEKpp_d
    ERKp_d = erktot - ERK_d - ERKpp_d

    # Sinapsis
    dCaMKII = p["kfck2"] * (powca / (powca + powkc)) - CaMKII / p["tauck2"]
    dRaf_s = -p["kfbasraf"] * Raf_s + p["kbraf"] * Rafp_s
    dMEK_s = -p["kfmkk"] * Rafp_s * MEK_s / (MEK_s + p["Kmkk"]) + p["kbmkk"] * MEKp_s / (MEKp_s + p["Kmkk"])
    dMEKpp_s = p["kfmkk"] * Rafp_s * MEKp_s / (MEKp_s + p["Kmkk"]) - p["kbmkk"] * MEKpp_s / (MEKpp_s + p["Kmkk"])
    dERK_s = -p["kferk"] * MEKpp_s * ERK_s / (ERK_s + p["Kmk"]) + p["kberk"] * ERKp_s / (ERKp_s + p["Kmk"])
    dERKpp_s = p["kferk"] * MEKpp_s * ERKp_s / (ERKp_s + p["Kmk"]) - p["kberk"] * ERKpp_s / (ERKpp_s + p["Kmk"])
    ERKact_s = ERKpp_s
    dPP = p["kfpp"] * (powca / (powca + powkc2)) - PP / p["taupp"]
    dtagp1 = p["kphos1"] * CaMKII * (1 - tagp1) - p["kdeph1"] * tagp1
    dtagd1 = p["kphos4"] * ERKact_s * (1 - tagd1) - p["kdeph4"] * tagd1
    dtagd2 = p["kdeph5"] * PP * (1 - tagd2) - p["kphos5"] * tagd2

    TLTP = tagp1**2
    TLTD = tagd1 * tagd2

    # Dendrita
    dCaMK_d = p["kfck2"] * (powcad / (powcad + powkcd)) - CaMK_d / p["tauck2"]
    dRaf_d = -p["kfbasraf"] * Raf_d + p["kbraf"] * Rafp_d
    dMEK_d = -p["kfmkk"] * Rafp_d * MEK_d / (MEK_d + p["Kmkk"]) + p["kbmkk"] * MEKp_d / (MEKp_d + p["Kmkk"])
    dMEKpp_d = p["kfmkk"] * MEKp_d * MEK_d / (MEK_d + p["Kmkk"]) - p["kbmkk"] * MEKpp_d / (MEKpp_d + p["Kmkk"])
    dERK_d = -p["kferk"] * MEKpp_d * ERK_d / (ERK_d + p["Kmk"]) + p["kberk"] * ERKp_d / (ERKp_d + p["Kmk"])
    dERKpp_d = p["kferk"] * MEKpp_d * ERKp_d / (ERKp_d + p["Kmk"]) - p["kberk"] * ERKpp_d / (ERKpp_d + p["Kmk"])
    ERKact_d = ERKpp_d
    dpsit1 = p["kphos7"] * ERKact_d * (1 - psit1) - p["kdeph7"] * psit1
    dpsit2 = p["kphos8"] * CaMK_d * (1 - psit2) - p["kdeph8"] * psit2
    dPRP = p["ktrans1"] * psit1**2 - PRP / p["tauprp"] + p["vtrbas1"]
    dPKM_d = p["kpkmon"] * psit1 * psit2 - PKM_d / p["taupkm"] + p["vtrbaspkm"] - p["kexpkm"] * TLTP * PKM_d + p["Vsd"] * p["klkpkm"] * PKM_s
    dPKM_s = p["kexpkm"] * TLTP * PKM_d / p["Vsd"] - p["klkpkm"] * PKM_s + p["vtrbaspkm"] + p["kpkmon2"] * PKM_s**2 / (PKM_s**2 + p["Kpkm"]**2) - PKM_s / p["taupkm"]
    dfsyn = p["kltp"] * PKM_s - fsyn / p["tfsyn"] + p["vfbas"]
    dnsyn = -p["kltd"] * TLTD * PRP * nsyn - nsyn / p["tnsyn"] + p["vdbas"]

    return [dCaMKII, dRaf_s, dMEK_s, dMEKpp_s, dERK_s, dERKpp_s, dPP, dtagp1,
            dtagd1, dtagd2, dCaMK_d, dRaf_d, dMEK_d, dMEKpp_d, dERK_d, dERKpp_d,
            dpsit1, dpsit2, dPRP, dPKM_d, dPKM_s, dfsyn, dnsyn]

# Condiciones iniciales
Y0 = [0.001, 0.125, 0.075, 0.1, 0.075, 0.1, 0.001, 0.001, 0.001, 0.001,
      0.001, 0.125, 0.075, 0.1, 0.075, 0.1, 0.001, 0.001, 0.001, 0.001,
      0.01, 0.01, 0.01]

# Resolver ODEs
sol = solve_ivp(smolen_odes, [t_start, t_end], Y0, method='BDF', t_eval=time_points, max_step=1.0)

# ==== Conversión a número de moléculas ====
var_names = [
    "CaMKII", "Raf_s", "MEK_s", "MEKpp_s", "ERK_s", "ERKpp_s", "PP", "tagp1",
    "tagd1", "tagd2", "CaMK_d", "Raf_d", "MEK_d", "MEKpp_d", "ERK_d", "ERKpp_d",
    "psit1", "psit2", "PRP", "PKM_d", "PKM_s", "fsyn", "nsyn"
]


syn_indices = list(range(10)) + [20, 21, 22]
dend_indices = list(range(10, 20))   # incluye PKM_d

mol_counts = []
for i, conc in enumerate(sol.y):
    volume = vol_syn if i in syn_indices else vol_dend
    mol = conc * 1e-6 * volume * NA
    mol_counts.append(mol)

""" 
import pickle

# Guardar resultados para el simulador híbrido
with open("mol_counts.pkl", "wb") as f:
    pickle.dump({
        "mol_counts": mol_counts,  # ← datos en número de moléculas
        "time": sol.t
    }, f)

print("\n[INFO] mol_counts.pkl guardado exitosamente.")

# ==== Graficar ====
plt.figure(figsize=(14, 8))
for i, label in enumerate(var_names):
    plt.plot(sol.t, mol_counts[i], label=label)

plt.xlabel("Time (min)")
plt.ylabel("Number of molecules")
plt.title("23-variable LTP/LTD Model (molecule counts)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("multiscale_model_molecules.png")

# Graficar cada variable en una figura separada
for i, label in enumerate(var_names):
    plt.figure(figsize=(6, 4))
    plt.plot(sol.t, mol_counts[i], label=label)
    plt.title(f"{label} (molecule count)")
    plt.xlabel("Time (min)")
    plt.ylabel("Number of molecules")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{label}_molecule_count.png")
    plt.close()

print("\n--- Estado estacionario (número de moléculas) ---")
for i, label in enumerate(var_names):
    molecules = mol_counts[i][-1]  # <- FIXED
    print(f"{label:10s} = {molecules:12,.0f}") """


