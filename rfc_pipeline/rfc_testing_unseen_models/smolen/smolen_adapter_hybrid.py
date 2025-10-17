# smolen_adapter_hybrid.py
# Usa smolen_odes.py como fuente (en µM), pero todo el híbrido corre en moléculas.
# Hace robusto el estímulo como ventana (o boxcar suave) con la MISMA duración efectiva (~0.1 min).

import numpy as np
import math
import importlib

# --- Carga el modelo original (NO se toca su RHS, solo se sobreescribe stimulus por una versión equivalente robusta)
S = importlib.import_module("smolen_odes")

# --- Constantes y volúmenes por especie (usa los del módulo si existen)
NA       = 6.022e23
VOL_SYN  = getattr(S, "vol_syn",  2.0e-16)  # L
VOL_DEND = getattr(S, "vol_dend", 2.0e-15)  # L

# --- Nombres y estado
var_names = list(getattr(S, "var_names"))
Y0_uM     = np.array(S.Y0, dtype=float)
nvars     = len(Y0_uM)
assert len(var_names) == nvars, "var_names debe coincidir con el tamaño del estado"

# --- Mapa de volúmenes por especie (ajústalo si tu orden difiere)
syn_indices  = list(range(10)) + [20,21,22]   # sinapsis: 0..9, PKM_s, fsyn, nsyn
dend_indices = list(range(10,20))             # dendrita: 10..19

def build_V_vector(n):
    V = np.empty(n, dtype=float)
    for i in range(n):
        V[i] = VOL_SYN if i in syn_indices else VOL_DEND
    return V

V    = build_V_vector(nvars)
Vcol = V.reshape(-1,1)

# --- Conversión µM <-> moléculas
def conc_to_mol(c_uM):
    # c[µM] * 1e-6 * V[L] * NA = moléculas
    if c_uM.ndim == 1:
        return c_uM * 1e-6 * V * NA
    return c_uM * 1e-6 * Vcol * NA

def mol_to_conc(n_mol):
    # n / (V*NA) * 1e6 = µM
    if n_mol.ndim == 1:
        return (n_mol / (V * NA)) * 1e6
    return (n_mol / (Vcol * NA)) * 1e6

def dconc_to_dmol(dc_uM_dt):
    if dc_uM_dt.ndim == 1:
        return dc_uM_dt * 1e-6 * V * NA
    return dc_uM_dt * 1e-6 * Vcol * NA

# --- Estímulo robusto (misma duración efectiva que tu script: ~0.1 min total)
#     Opción A: ventana rectangular corta (área similar a abs(t-stim)<0.05)
PULSE_WIDTH = 0.1   # minutos (ajústalo si tu script usa otro equivalente)
Cabas       = getattr(S, "Cabas", 0.04)
AMPTETCA    = getattr(S, "AMPTETCA", 1.4)
AMPTETCAD   = getattr(S, "AMPTETCAD", 0.65)  # tu valor

stim_times  = list(getattr(S, "stim_times", []))

USE_SMOOTH_BOXCAR = False  # pon True si prefieres bordes suaves (mismo área aprox.)
EPS_EDGE = 0.02            # ancho de suavizado de borde (si se usa boxcar suave)

def _level_rect(t):
    # 1 si t ∈ [s, s+PULSE_WIDTH), 0 fuera (para cualquier pulso)
    for s in stim_times:
        if s <= t < s + PULSE_WIDTH:
            return 1.0
    return 0.0

def _level_smooth(t):
    # Máximo de boxcars suavizados para todos los pulsos
    lev = 0.0
    for s in stim_times:
        a = 0.5*(math.tanh((t - s)/EPS_EDGE) + 1.0)
        b = 0.5*(math.tanh((t - (s + PULSE_WIDTH))/EPS_EDGE) + 1.0)
        lev = max(lev, a - b)
    return lev

def stimulus_robust(t):
    if not stim_times:
        return Cabas, Cabas
    level = _level_smooth(t) if USE_SMOOTH_BOXCAR else _level_rect(t)
    casyn  = Cabas + (AMPTETCA  - Cabas) * level
    cadend = Cabas + (AMPTETCAD - Cabas) * level
    return casyn, cadend

# --- Envolvemos la RHS original para usar el estímulo robusto
#     Creamos un closure que llama S.smolen_odes pero sustituyendo stimulus por la robusta.
_original_stimulus = S.stimulus if hasattr(S, "stimulus") else None

def smolen_odes_with_robust_stimulus(t, y_uM):
    # Monkey-patch temporal del estímulo del módulo
    if _original_stimulus is not None:
        S.stimulus = stimulus_robust
    dy = S.smolen_odes(t, y_uM)
    # Restaurar por limpieza (opcional en una sola llamada)
    if _original_stimulus is not None:
        S.stimulus = _original_stimulus
    return dy

# --- ODE en moléculas para el híbrido
def smolen_odes_mol(t, y_mol):
    y_uM   = mol_to_conc(np.array(y_mol, dtype=float))
    dy_uM  = np.array(smolen_odes_with_robust_stimulus(t, y_uM), dtype=float)  # µM/min
    dy_mol = dconc_to_dmol(dy_uM)                                              # moléculas/min
    return dy_mol

# --- Y0 en moléculas y params “reexportados”
Y0     = conc_to_mol(Y0_uM)
params = getattr(S, "params", {})

# --- Sugerencia para el solver determinista en el híbrido:
#     usa method='BDF' y max_step <= min(0.02, DT) si los pulsos son de 0.1 min.
