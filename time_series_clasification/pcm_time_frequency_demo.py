#!/usr/bin/env python3
"""
Caracterización microarquitectónica de cargas de trabajo con PCM — Random Forest
================================================================================

Versión SUPERVISADA del pipeline. En lugar de PCA + K-means (que nunca ve las
etiquetas), aquí se entrena un Random Forest que SÍ aprende de las etiquetas,
siguiendo los pasos de "A Practical Guide to Implementing a Random Forest
Classifier in Python" (Eden Molina, TDS, 2021), adaptados a series de tiempo.

¿Qué hace este script, en pocas palabras?
-----------------------------------------
1. LEE los CSV de Intel PCM para cada aplicación (7zip, ffmpeg, ..., cryptsky,
   ransomware) y se queda con UNA corrida de cada una (PCM_RUN, por defecto la 1).
2. CALCULA 6 métricas por muestra: CPI, IPC, LLC_MPKI, L2_MPKI, L2_HIT y L3_HIT.
3. CORTA cada corrida en VENTANAS de 1, 2, 5, 10, 15 y 20 segundos.
4. Para cada ventana EXTRAE features (tiempo, frecuencia FFT e híbrido).
5. SEPARA en el tiempo, dentro de cada corrida: el primer 80 % de las ventanas
   es entrenamiento y el último 20 % es prueba (split temporal). No se reparte
   al azar: dos ventanas vecinas son casi idénticas y el modelo "se copiaría".
6. ENTRENA un Random Forest con el 80 % y lo evalúa con el 20 %: precision,
   recall, F1 (por clase y maligno vs. benigno), ROC AUC, sobreajuste
   (train vs. test), importancia de features y una prueba de generalización:
   ¿detecta un ransomware que NUNCA vio al entrenar?
7. GRAFICA las 6 métricas por pares (15 pares, un color por workload) y ordena
   los pares según qué tan bien agrupan a todos los malignos (11_*, 12_*).
   Todo se guarda en ./pcm_rf_output_temporal_run1/window_XXs/ y la tabla
   comparativa en ./pcm_rf_output_temporal_run1/stats_all_windows.md

Dependencias:
    pip install numpy pandas matplotlib scikit-learn scipy
    pip install adjustText      # opcional: acomoda los nombres en las gráficas por pares

Ejecutar:
    python pcm_time_frequency_rf.py                     # corrida 1, split 80/20
    PCM_RUN=2 python pcm_time_frequency_rf.py           # usar la corrida 2
    PCM_RUN=all python pcm_time_frequency_rf.py         # todas, cada una 80/20
    PCM_TEST_FRAC=0.3 python pcm_time_frequency_rf.py   # split 70/30
    PCM_DATA_DIR=/ruta/a/Resultados_ALL python pcm_time_frequency_rf.py
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np                  # cálculo numérico (arreglos, FFT)
import pandas as pd                 # tablas (DataFrames) y lectura de CSV
import matplotlib
# Backend "Agg": dibuja directo a archivo, sin ventanas. Sin esto, matplotlib usa
# Tkinter, que choca con los hilos del Random Forest (n_jobs=-1) y el programa
# truena con "Tcl_AsyncDelete: async handler deleted by the wrong thread".
# También permite correr el script en una máquina sin interfaz gráfica.
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # gráficas

from sklearn.preprocessing import StandardScaler   # solo para la gráfica PCA de visualización
from sklearn.decomposition import PCA              # solo para la gráfica PCA de visualización
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score, classification_report,
                             balanced_accuracy_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors     # solo para medir traslape en las gráficas por pares
from itertools import combinations
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

try:   # opcional (pip install adjustText): evita que se encimen los nombres de las clases
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


# Generador de números aleatorios con semilla fija → resultados reproducibles.
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------
# Configuración de los datos
# ---------------------------------------------------------------------
# Carpeta con los resultados. Se espera una subcarpeta por clase, por ejemplo:
#   ~/Desktop/Research/Resultados_ALL/7zip/xxx_pcm_1.csv
#   ~/Desktop/Research/Resultados_ALL/cryptsky/xxx_pcm_1.csv
# Se puede cambiar sin tocar el código con la variable de entorno PCM_DATA_DIR.
DATA_DIR = Path(os.environ.get("PCM_DATA_DIR",
                               Path.home() / "Desktop" / "Research" / "Resultados_ALL"))

FS = 10.0                  # Frecuencia de muestreo: 10 muestras por segundo (PCM cada 0.1 s)

# Tamaños de ventana (en segundos) a evaluar.
WINDOW_SIZES_S = [1, 2, 5, 10, 15, 20]

# Ventana "actual" (main() las cambia con set_window()).
WINDOW_S = 20
WINDOW_N = int(FS * WINDOW_S)

METRICS = ["CPI", "IPC", "LLC_MPKI", "L2_MPKI", "L2_HIT", "L3_HIT"]

CLASSES = ["7zip", "ffmpeg", "gpg", "openssl", "readwrite", "rsync", "stress",
           "sysbench", "mlc", "cryptsky", "ransomware", "trojan", "ransim"]

MALICIOUS = ["cryptsky", "ransomware", "trojan", "ransim"]

# Columnas que identifican una ventana (NO son features).
META_COLS = ["pid", "class", "window", "start_s"]

# Colores fijos por clase (benignos: tab10 sin rojo; malignos: rojo, negro, ...).
_BENIGN_PALETTE = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 3]
_MAL_PALETTE = ["#d62728", "#000000", "#8b0000", "#4b0082"]
COLORS = {}
for _c in CLASSES:
    if _c in MALICIOUS:
        COLORS[_c] = _MAL_PALETTE[MALICIOUS.index(_c) % len(_MAL_PALETTE)]
    else:
        _i = len([k for k in COLORS if k not in MALICIOUS])
        COLORS[_c] = _BENIGN_PALETTE[_i % len(_BENIGN_PALETTE)]

# Marcador por clase maligna en las gráficas por pares: rojo y rojo oscuro se
# confunden, así que cada familia lleva además su propia forma.
_MAL_MARKERS = ["X", "P", "^", "D"]
MARKERS = {c: (_MAL_MARKERS[MALICIOUS.index(c) % len(_MAL_MARKERS)] if c in MALICIOUS else "o")
           for c in CLASSES}

# ---------------------------------------------------------------------
# Separación entrenamiento / prueba: split TEMPORAL dentro de cada corrida
# ---------------------------------------------------------------------
# Cada corrida se ordena en el tiempo; el primer (1 - TEST_FRAC) de sus ventanas
# entrena y el último TEST_FRAC prueba. Responde: "si el modelo ya vio el inicio
# de una corrida, ¿reconoce el resto?". Ojo: como entrenamiento y prueba salen
# de la misma corrida, NO mide si el modelo generaliza a corridas nuevas.
#   PCM_TEST_FRAC → fracción final de cada corrida usada como prueba (0.2 = 20 %).
#   PCM_RUN       → qué corrida de cada workload se usa:
#                   "1" (por defecto) = solo la primera, "2" = solo la segunda, ...,
#                   "all" = todas (cada una partida 80/20 por separado).
TEST_FRAC = float(os.environ.get("PCM_TEST_FRAC", "0.2"))
RUN_CHOICE = os.environ.get("PCM_RUN", "1")
#   PCM_TARGET_FPR → tasa de falsas alarmas aceptada al calibrar el umbral de la
#                    prueba "ransomware nunca visto" (0.01 = 1 % de ventanas benignas).
TARGET_FPR = float(os.environ.get("PCM_TARGET_FPR", "0.01"))
if not 0 < TEST_FRAC < 1:
    raise ValueError("PCM_TEST_FRAC debe estar entre 0 y 1")
if not 0 < TARGET_FPR < 1:
    raise ValueError("PCM_TARGET_FPR debe estar entre 0 y 1")
if RUN_CHOICE != "all" and not RUN_CHOICE.isdigit():
    raise ValueError("PCM_RUN debe ser un número (1, 2, ...) o 'all'")

# Carpeta de salida: una por corrida elegida, para no mezclar resultados.
BASE_OUT = Path(f"pcm_rf_output_temporal_run{RUN_CHOICE}")
OUT = BASE_OUT

# ---------------------------------------------------------------------
# Configuración del Random Forest
# ---------------------------------------------------------------------
# Parámetros del bosque cuando NO se hace búsqueda de hiperparámetros.
# Los valores por defecto de sklearn ya funcionan bien; solo se agregan:
#   class_weight="balanced" → si una clase tiene más ventanas que otra,
#                             se compensa para que no domine el entrenamiento.
#   min_samples_leaf=2      → hojas con al menos 2 ventanas: reduce un poco
#                             el sobreajuste que el artículo observa (train = 100%).
RF_PARAMS = dict(
    n_estimators=300,          # número de árboles
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,                 # usa todos los núcleos
    random_state=42,
)

# Búsqueda aleatoria de hiperparámetros (RandomizedSearchCV, como en el artículo).
# Es lenta: actívala cuando ya hayas elegido un tamaño de ventana, p. ej.
# poniendo WINDOW_SIZES_S = [10] y TUNE = True.
# La búsqueda usa SOLO el 80 % de entrenamiento (dentro de él vuelve a separar
# en el tiempo: inicio para entrenar, final para validar), para no "espiar"
# el 20 % de prueba.
TUNE = False
TUNE_ITER = 20             # combinaciones aleatorias a probar (el artículo usa 30)
# Rango de búsqueda. Respecto al artículo se corrigen dos cosas que en
# scikit-learn actual truenan: max_features="auto" ya no existe y
# min_samples_split debe ser >= 2.
PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500, 800],
    "max_depth": [None, 5, 10, 20, 40],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
}


def set_window(window_s: int):
    """Fija la ventana actual (WINDOW_S, WINDOW_N) y la carpeta de salida OUT."""
    global WINDOW_S, WINDOW_N, OUT
    WINDOW_S = window_s
    WINDOW_N = int(FS * window_s)
    OUT = BASE_OUT / f"window_{window_s:02d}s"
    OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Carga de la telemetría de PCM  (sin cambios respecto a la versión K-means)
# ---------------------------------------------------------------------
def load_pcm_csv(path: Path, label: str, run_id: int) -> pd.DataFrame:
    """Lee un CSV de pcm (2 filas de encabezado) y devuelve el formato largo,
    usando solo el grupo "System" (totales de toda la máquina)."""
    df = pd.read_csv(path, header=[0, 1])
    sysd = df["System"]

    def col(name):
        return pd.to_numeric(sysd[name], errors="coerce")

    ipc = col("IPC")
    out = pd.DataFrame({
        "IPC": ipc,
        "CPI": 1.0 / ipc.where(ipc > 0),
        "LLC_MPKI": col("L3MISS") / col("INST") * 1000.0,
        "L2_MPKI": col("L2MISS") / col("INST") * 1000.0,
        "L2_HIT": col("L2HIT"),
        "L3_HIT": col("L3HIT"),

    })

    # Interpolar huecos para no romper el espaciado de 0.1 s (importante para la FFT).
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.interpolate(limit_direction="both")

    out.insert(0, "time_s", np.arange(len(out)) / FS)
    out.insert(1, "pid", run_id)          # "pid" = id de corrida (no el PID del proceso)
    out.insert(2, "class", label)
    out["run_file"] = path.name
    return out


def load_pcm_dataset(min_samples: int) -> pd.DataFrame:
    """Lee todos los CSV de todas las clases. Omite corridas con < min_samples muestras."""
    frames = []
    run_id = 1000
    for label in CLASSES:
        files = sorted((DATA_DIR / label).glob("*pcm*.csv"))
        if not files:
            print(f"[aviso] sin CSV para '{label}' en {DATA_DIR / label}")
            continue
        for f in files:
            d = load_pcm_csv(f, label, run_id)
            if len(d) < min_samples:
                print(f"[aviso] {f.name}: {len(d)} muestras (< {min_samples}), se omite")
                continue
            frames.append(d)
            run_id += 1
    return pd.concat(frames, ignore_index=True)


def first_pid(raw: pd.DataFrame, label: str):
    """Id de la primera corrida de una clase, o None si no hay datos."""
    m = raw["class"] == label
    return raw.loc[m, "pid"].iloc[0] if m.any() else None


# ---------------------------------------------------------------------
# Ventaneo y extracción de features  (sin cambios)
# ---------------------------------------------------------------------
def iter_windows(df: pd.DataFrame):
    """Corta cada corrida en ventanas consecutivas, sin traslape, de WINDOW_N muestras."""
    for (pid, label), g in df.groupby(["pid", "class"], sort=False):
        g = g.sort_values("time_s").reset_index(drop=True)
        n_windows = len(g) // WINDOW_N
        for w in range(n_windows):
            start = w * WINDOW_N
            win = g.iloc[start:start + WINDOW_N].copy()
            yield pid, label, w, win


def spectral_features(x: np.ndarray, fs: float) -> dict:
    """peak_freq, peak_power, spectral_energy y spectral_entropy de una señal (vía FFT)."""
    x = np.asarray(x, dtype=float)
    centered = x - x.mean()
    fft = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0/fs)
    power = np.abs(fft) ** 2

    freqs_nd = freqs[1:]
    power_nd = power[1:]
    total_power = power_nd.sum()
    if total_power <= 1e-15:
        return {"peak_freq": 0.0, "peak_power": 0.0,
                "spectral_energy": 0.0, "spectral_entropy": 0.0}

    peak_idx = np.argmax(power_nd)
    p = power_nd / total_power
    entropy = -(p * np.log2(p + 1e-15)).sum()
    return {
        "peak_freq": float(freqs_nd[peak_idx]),
        "peak_power": float(power_nd[peak_idx]),
        "spectral_energy": float(total_power),
        "spectral_entropy": float(entropy),
    }


def extract_features(raw: pd.DataFrame):
    """Una fila por ventana. Devuelve time_df (8 features), freq_df (16) y hybrid_df (24)."""
    time_rows, freq_rows = [], []
    for pid, label, w, win in iter_windows(raw):
        meta = {"pid": pid, "class": label, "window": w,
                "start_s": float(win["time_s"].iloc[0])}
        trow, frow = dict(meta), dict(meta)
        for metric in METRICS:
            x = win[metric].to_numpy()
            trow[f"{metric}_mean"] = float(np.mean(x))
            trow[f"{metric}_std"] = float(np.std(x))
            for name, value in spectral_features(x, FS).items():
                frow[f"{metric}_{name}"] = value
        time_rows.append(trow)
        freq_rows.append(frow)

    time_df = pd.DataFrame(time_rows)
    freq_df = pd.DataFrame(freq_rows)
    hybrid_df = time_df.merge(freq_df, on=META_COLS, how="inner")
    return time_df, freq_df, hybrid_df


# ---------------------------------------------------------------------
# Gráficas explicativas  (sin cambios)
# ---------------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_raw_examples(raw):
    """01 y 02: serie de tiempo de CPI y LLC_MPKI (primera corrida de cada clase)."""
    for metric, title, ylabel, fname in [
        ("CPI", "PCM CPI time series (System)", "CPI", "01_raw_cpi_timeseries.png"),
        ("LLC_MPKI", "PCM LLC MPKI time series (System)", "LLC MPKI", "02_raw_llc_mpki_timeseries.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for label in CLASSES:
            pid = first_pid(raw, label)
            if pid is None:
                continue
            g = raw[raw["pid"] == pid]
            ax.plot(g["time_s"], g[metric], label=f"{label} (run {pid})",
                    color=COLORS[label],
                    linewidth=2.0 if label in MALICIOUS else 1.2)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(alpha=0.25)
        savefig(fname)


def plot_window_example(raw):
    """03: una ventana de CPI de cada clase maligna y su media."""
    for label in MALICIOUS:
        pid = first_pid(raw, label)
        if pid is None:
            continue
        g = raw[raw["pid"] == pid].sort_values("time_s").reset_index(drop=True)
        win = g.iloc[:WINDOW_N]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(win["time_s"], win["CPI"], marker="o", markersize=2, color=COLORS[label])
        ax.axhline(win["CPI"].mean(), linestyle="--",
                   label=f"mean={win['CPI'].mean():.3f}")
        ax.set_title(f"One {WINDOW_S}-second CPI window — {label} (run {pid})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CPI")
        ax.legend()
        ax.grid(alpha=0.25)
        savefig(f"03_single_window_mean_{label}.png")


def plot_same_mean_different_structure():
    """04 y 05 (datos inventados): misma media/std, distinta estructura en frecuencia."""
    n = WINDOW_N
    t = np.arange(n) / FS
    a = 1.0 + 0.18*np.sin(2*np.pi*0.4*t)
    b = 1.0 + RNG.normal(0, np.std(a), n)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, a, label=f"Periodic: mean={a.mean():.2f}, std={a.std():.2f}")
    ax.plot(t, b, label=f"Noisy: mean={b.mean():.2f}, std={b.std():.2f}", alpha=0.8)
    ax.set_title("Similar time-domain statistics, different temporal structure")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Illustrative CPI")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("04_similar_stats_different_structure.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for x, label in [(a, "Periodic"), (b, "Noisy")]:
        centered = x - x.mean()
        freq = np.fft.rfftfreq(n, 1/FS)
        power = np.abs(np.fft.rfft(centered)) ** 2
        ax.plot(freq[1:], power[1:], label=label)
    ax.set_title("FFT exposes the difference")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("05_fft_periodic_vs_noise.png")


def plot_fft_examples(raw):
    """06: espectro de potencia del CPI en la primera ventana de cada clase."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in CLASSES:
        pid = first_pid(raw, label)
        if pid is None:
            continue
        g = raw[raw["pid"] == pid].sort_values("time_s").reset_index(drop=True)
        x = g.iloc[:WINDOW_N]["CPI"].to_numpy()
        centered = x - x.mean()
        freq = np.fft.rfftfreq(len(x), 1/FS)
        power = np.abs(np.fft.rfft(centered)) ** 2
        ax.plot(freq[1:], power[1:], label=label, color=COLORS[label],
                linewidth=2.0 if label in MALICIOUS else 1.2)
    ax.set_title(f"CPI power spectra for one {WINDOW_S}-second window")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_yscale("log")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.25)
    savefig("06_cpi_power_spectra.png")


def pca_plot(df, title, filename):
    """07: PCA 2D SOLO PARA VISUALIZAR. El Random Forest no usa PCA."""
    feature_cols = feature_columns(df)
    Xz = StandardScaler().fit_transform(df[feature_cols].to_numpy())
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xz)

    pca_df = df[META_COLS].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]
    pca_df.to_csv(OUT / filename.replace(".png", ".csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASSES:
        m = pca_df["class"] == label
        if not m.any():
            continue
        is_rw = label in MALICIOUS
        ax.scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"],
                   label=label, color=COLORS[label],
                   marker="X" if is_rw else "o",
                   s=120 if is_rw else 35,
                   edgecolors=("white" if COLORS[label] == "#000000" else "black") if is_rw else "none",
                   alpha=0.9 if is_rw else 0.6,
                   zorder=3 if is_rw else 2)
    evr = pca.explained_variance_ratio_ * 100
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({evr[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1f}% variance)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.25)
    savefig(filename)


# ---------------------------------------------------------------------
# Gráficas por pares de métricas (11 y 12)
# ---------------------------------------------------------------------
# Objetivo: buscar un par de métricas en el que TODOS los malignos queden
# agrupados (juntos entre sí y lejos de los benignos). Cada punto es una ventana
# (la media de la métrica en la ventana, igual que las features *_mean).
#
# Escalas: las MPKI van de ~0.005 (cryptsky) a ~25 (mlc); en lineal todo queda
# aplastado contra el cero. "symlog" es logarítmica pero admite el 0 (LLC_MPKI
# tiene muestras en 0).
PAIR_SCALES = {
    "CPI": ("log", {}),
    "IPC": ("linear", {}),
    "LLC_MPKI": ("symlog", {"linthresh": 0.01}),
    "L2_MPKI": ("symlog", {"linthresh": 0.01}),
    "L2_HIT": ("linear", {}),
    "L3_HIT": ("linear", {}),
}
PAIR_LABELS = {"CPI": "CPI", "IPC": "IPC", "LLC_MPKI": "LLC MPKI",
               "L2_MPKI": "L2 MPKI", "L2_HIT": "L2 hit ratio", "L3_HIT": "L3 hit ratio"}
# Métricas que se pasan a log10 antes de medir distancias (vecinos más cercanos).
_LOG_METRICS = {"CPI", "LLC_MPKI", "L2_MPKI"}


def _tick_fmt(v, _pos=None):
    return "0" if v == 0 else f"{v:g}"


def _set_pair_scale(ax, which, metric):
    """Aplica la escala de la métrica a un eje ('x' o 'y') con ticks legibles."""
    scale, kw = PAIR_SCALES[metric]
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)(scale, **kw)
    if scale == "symlog":
        if which == "x":
            ax.set_xlim(left=0)          # las MPKI nunca son negativas
        else:
            ax.set_ylim(bottom=0)
        axis.set_major_locator(FixedLocator([0, 0.01, 0.1, 1, 10, 100]))
    elif scale == "log":
        axis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 5, 10]))
    if scale != "linear":
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(_tick_fmt))


def _pair_scatter(ax, w, x, y, small=False):
    """Dispersión de un par: un color por workload; malignos con su propio marcador."""
    for label in CLASSES:
        d = w[w["class"] == label]
        if d.empty:
            continue
        mal = label in MALICIOUS
        ax.scatter(d[x], d[y], color=COLORS[label], marker=MARKERS[label],
                   s=(14 if small else 40) if mal else (4 if small else 12),
                   alpha=0.85 if mal else 0.45, linewidths=0.4 if mal else 0,
                   edgecolors="white" if mal else "none",
                   zorder=3 if mal else 2, label=label, rasterized=True)
    _set_pair_scale(ax, "x", x)
    _set_pair_scale(ax, "y", y)
    ax.grid(alpha=0.25)


def _class_legend_handles():
    return [plt.Line2D([], [], ls="", marker=MARKERS[c], color=COLORS[c],
                       markersize=8 if c in MALICIOUS else 6,
                       label=c + (" (maligno)" if c in MALICIOUS else ""))
            for c in CLASSES]


def pair_separability(w: pd.DataFrame) -> pd.DataFrame:
    """¿En qué par de métricas quedan TODOS los malignos agrupados?

    Para cada par (2 features: las medias de la ventana) se mide:
      silhouette    → silhouette de las etiquetas maligno/benigno (-1 a 1). Cerca
                      de 1 = los malignos forman un grupo compacto y separado;
                      cerca de 0 = se enciman con los benignos.
      radio_mal     → distancia de la familia maligna más alejada al centro de
                      los malignos (en z-score). Chico = las familias están juntas.
      benigno_mas_cerca / dist_benigno → el benigno cuya mediana queda más cerca
                      de la mediana de alguna familia maligna (quién "invade").
      det_no_visto  → detección promedio de una familia maligna que el bosque
                      NUNCA vio (umbral 0.5). Si los malignos forman un grupo, la
                      familia nueva cae junto a las otras y se detecta.
      mal_mezclado  → % de ventanas malignas cuyos 10 vecinos más cercanos son en
                      su mayoría benignos.
      bal_acc       → bosque binario (maligno vs. benigno) con el mismo split
                      temporal 80/20; exactitud balanceada en prueba.
    Las distancias usan log10 en CPI/MPKI y luego z-score.
    """
    w = w.reset_index(drop=True)
    cls = w["class"].to_numpy()
    yb = np.isin(cls, MALICIOUS).astype(int)
    mal_present = [c for c in MALICIOUS if c in set(cls)]
    tr, te = temporal_split(w, TEST_FRAC)
    is_tr = np.zeros(len(w), bool)
    is_tr[tr] = True
    # Escala común para las distancias: log10 en CPI/MPKI y luego z-score.
    Z_all = pd.DataFrame({m: np.log10(w[m] + 1e-3) if m in _LOG_METRICS else w[m]
                          for m in METRICS})
    Z_all = (Z_all - Z_all.mean()) / Z_all.std()
    params = {**RF_PARAMS, "n_estimators": 100}   # 100 árboles bastan para 2 features
    rows = []
    for x, y in combinations(METRICS, 2):
        X = w[[x, y]].to_numpy()
        row = {"par": f"{y} vs {x}"}
        clf = RandomForestClassifier(**params).fit(X[tr], yb[tr])
        row["bal_acc"] = balanced_accuracy_score(yb[te], clf.predict(X[te]))
        dets = []
        for fam in mal_present:
            tr_f = is_tr & (cls != fam)
            te_f = ~is_tr & ((yb == 0) | (cls == fam))
            clf_f = RandomForestClassifier(**params).fit(X[tr_f], yb[tr_f])
            det = recall_score((cls[te_f] == fam).astype(int), clf_f.predict(X[te_f]),
                               zero_division=0)
            row[f"det {fam} no visto"] = det
            dets.append(det)
        row["det_no_visto"] = float(np.mean(dets)) if dets else np.nan
        Z = Z_all[[x, y]].to_numpy()
        k = min(10, len(Z) - 1)
        _, ix = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z[yb == 1])
        row["mal_mezclado"] = float((yb[ix[:, 1:]].mean(axis=1) < 0.5).mean())
        row["silhouette"] = float(silhouette_score(Z, yb, sample_size=min(2000, len(Z)),
                                                   random_state=42))
        med = pd.DataFrame(Z, columns=["a", "b"]).groupby(cls).median()
        center = med.loc[mal_present].to_numpy().mean(axis=0)
        row["radio_mal"] = float(np.linalg.norm(med.loc[mal_present].to_numpy() - center,
                                                axis=1).max())
        # Benigno más cercano a CUALQUIER familia maligna (no al centro: cryptsky
        # jala el centro hacia una zona vacía).
        ben = med.drop(index=mal_present)
        mal_med = med.loc[mal_present].to_numpy()
        dist = pd.Series(np.linalg.norm(ben.to_numpy()[:, None, :] - mal_med[None, :, :],
                                        axis=2).min(axis=1), index=ben.index)
        row["benigno_mas_cerca"] = dist.idxmin()
        row["dist_benigno"] = float(dist.min())
        rows.append(row)
    cols = ["par", "silhouette", "radio_mal", "benigno_mas_cerca", "dist_benigno",
            "det_no_visto", "mal_mezclado", "bal_acc"] + \
           [f"det {f} no visto" for f in mal_present]
    return pd.DataFrame(rows)[cols].sort_values("silhouette", ascending=False)


def plot_metric_pairs(time_df: pd.DataFrame) -> list:
    """11 y 12: las 6 métricas comparadas por pares (15 pares), un color por workload.

      11_pairs_grid.png       matriz 6x6: dispersión de cada par, distribución de
                              cada métrica en la diagonal y ρ de Spearman arriba.
      12_pairs/<y>_vs_<x>.png un archivo por par, con el nombre de cada workload
                              en su mediana.
      12_pairs_separability.csv  ranking de pares (ver pair_separability()).
    Devuelve líneas de Markdown para stats.md.
    """
    w = time_df[META_COLS].copy()
    for m in METRICS:
        w[m] = time_df[f"{m}_mean"]
    pairs = list(combinations(METRICS, 2))
    k = len(METRICS)
    rho = w[METRICS].corr(method="spearman")

    # --- 11: matriz 6x6 ---
    fig, axes = plt.subplots(k, k, figsize=(18, 16))
    for i, yname in enumerate(METRICS):
        for j, xname in enumerate(METRICS):
            ax = axes[i, j]
            if j > i:                      # triángulo superior: correlación
                ax.axis("off")
                ax.text(0.5, 0.55, f"ρ = {rho.loc[yname, xname]:+.2f}", ha="center",
                        va="center", fontsize=14, transform=ax.transAxes)
                ax.text(0.5, 0.36, f"{PAIR_LABELS[yname]} · {PAIR_LABELS[xname]}",
                        ha="center", va="center", fontsize=8, color="0.35",
                        transform=ax.transAxes)
                continue
            if i == j:                     # diagonal: distribución por workload
                vals = w[xname]
                scale = PAIR_SCALES[xname][0]
                if scale == "log":
                    bins = np.logspace(np.log10(max(vals.min(), 1e-3)), np.log10(vals.max()), 40)
                elif scale == "symlog":
                    bins = np.concatenate([np.linspace(0, 0.01, 5)[:-1],
                                           np.logspace(-2, np.log10(max(vals.max(), 0.02)), 36)])
                else:
                    bins = np.linspace(vals.min(), vals.max(), 40)
                for label in CLASSES:
                    d = w.loc[w["class"] == label, xname]
                    if not d.empty:
                        ax.hist(d, bins=bins, histtype="step", color=COLORS[label],
                                linewidth=1.6 if label in MALICIOUS else 1.0)
                _set_pair_scale(ax, "x", xname)
                ax.set_yticks([])
            else:
                _pair_scatter(ax, w, xname, yname, small=True)
            ax.tick_params(labelsize=7)
            if i == k - 1:
                ax.set_xlabel(PAIR_LABELS[xname])
            if j == 0 and i != 0:
                ax.set_ylabel(PAIR_LABELS[yname])
    fig.legend(handles=_class_legend_handles(), loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"PCM metrics, pairwise ({WINDOW_S}-second window means)", fontsize=15)
    savefig("11_pairs_grid.png")

    # --- 12: un archivo por par, con etiquetas en la mediana de cada workload ---
    pair_dir = OUT / "12_pairs"
    pair_dir.mkdir(exist_ok=True)
    med = w.groupby("class")[METRICS].median()
    for x, y in pairs:
        fig, ax = plt.subplots(figsize=(9, 7))
        _pair_scatter(ax, w, x, y)
        texts = [ax.text(med.loc[c, x], med.loc[c, y], c, fontsize=8,
                         fontweight="bold" if c in MALICIOUS else "normal",
                         color=COLORS[c], zorder=5,
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
                 for c in CLASSES if c in med.index]
        if adjust_text is not None:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5))
        ax.set_xlabel(PAIR_LABELS[x])
        ax.set_ylabel(PAIR_LABELS[y])
        ax.set_title(f"{PAIR_LABELS[y]} vs {PAIR_LABELS[x]} — {WINDOW_S}-second window means")
        ax.legend(handles=_class_legend_handles(), loc="upper left",
                  bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(pair_dir / f"{y}_vs_{x}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Ranking: ¿algún par agrupa a todos los malignos? ---
    print("Pairwise separability...")
    sep = pair_separability(w)
    sep.to_csv(OUT / "12_pairs_separability.csv", index=False)
    rho.round(3).to_csv(OUT / "11_pairs_spearman.csv")
    print(sep.round(3).to_string(index=False))
    return [
        "## Pares de métricas: ¿algún par agrupa a todos los malignos?\n",
        "Cada par usa solo 2 features (las medias de la ventana). Ordenado por "
        "`silhouette` maligno vs. benigno (1 = malignos en un grupo compacto y aparte; "
        "0 = encimados con benignos). `radio_mal` = distancia de la familia maligna más "
        "alejada al centro maligno (z-score; chico = familias juntas). "
        "`benigno_mas_cerca` / `dist_benigno` = benigno más cercano a alguna familia maligna. "
        "`det_no_visto` = detección promedio de una familia que el bosque nunca vio "
        "(si los malignos forman grupo, cae junto a las otras). `mal_mezclado` = % de "
        "ventanas malignas cuyos 10 vecinos son mayormente benignos. Gráficas en "
        "`11_pairs_grid.png` y `12_pairs/`.\n",
        md_table(list(sep.columns), sep.values.tolist()),
        "",
    ]


def md_table(header, rows) -> str:
    """Tabla en Markdown hecha a mano (para no depender de 'tabulate')."""
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Random Forest: utilidades
# ---------------------------------------------------------------------
def feature_columns(df):
    """Todas las columnas que no son de identificación."""
    return [c for c in df.columns if c not in META_COLS]


def temporal_split(df, test_frac):
    """Split DENTRO de cada corrida, respetando el orden del tiempo.

    Para cada corrida, sus ventanas se ordenan por tiempo; el primer
    (1 - test_frac) va a entrenamiento y el último test_frac a prueba.
    Ejemplo con ventana de 10 s y una corrida de ~300 s (30 ventanas):
        ventanas 0–23 (0–240 s)    → entrenamiento
        ventanas 24–29 (240–300 s) → prueba
    No se reparte al azar a propósito: dos ventanas vecinas son casi idénticas,
    y si se mezclan el modelo acierta por memoria. Cortando en el tiempo solo
    la ventana justo antes del corte y la justo después son "vecinas".

    Devuelve (índices_entrenamiento, índices_prueba).
    """
    idx = np.arange(len(df))
    is_test = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby("pid", sort=False):
        order = g.sort_values("window").index.to_numpy()
        pos = df.index.get_indexer(order)
        n_test = max(1, int(round(len(pos) * test_frac)))
        is_test[pos[-n_test:]] = True
    return idx[~is_test], idx[is_test]


def inner_temporal_split(groups, frac=0.25):
    """Split de validación para la búsqueda de hiperparámetros (TUNE=True).

    Recibe solo las ventanas de ENTRENAMIENTO (ya en orden de tiempo dentro de
    cada corrida) y vuelve a cortar en el tiempo: inicio → entrena, final → valida.
    Devuelve una lista con un (índices_entrena, índices_valida), formato de `cv`.
    """
    groups = np.asarray(groups)
    is_val = np.zeros(len(groups), dtype=bool)
    for g in np.unique(groups):
        pos = np.flatnonzero(groups == g)
        n_val = max(1, int(round(len(pos) * frac)))
        is_val[pos[-n_val:]] = True
    idx = np.arange(len(groups))
    return [(idx[~is_val], idx[is_val])]


def check_split(tr, te, y, classes):
    """Verificación de seguridad: avisa si alguna clase falta en entrenamiento
    o en prueba (el síntoma sería recall = 0.000 en esa clase)."""
    ok = True
    for c in range(len(classes)):
        if c not in set(y[tr]):
            print(f"[ERROR] '{classes[c]}' no tiene ventanas de ENTRENAMIENTO")
            ok = False
        if c not in set(y[te]):
            print(f"[aviso] '{classes[c]}' no tiene ventanas de PRUEBA")
    return ok


def build_model(X_tr, y_tr, g_tr, seed=42):
    """Entrena el Random Forest con las ventanas de entrenamiento.

    TUNE=False → parámetros fijos (RF_PARAMS), equivalente al
                 `clf = RandomForestClassifier(); clf.fit(...)` del artículo.
    TUNE=True  → RandomizedSearchCV como en el artículo; valida con el final
                 (en el tiempo) de las ventanas de entrenamiento.
    Devuelve (modelo_entrenado, mejores_parámetros o None).
    """
    base = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
    if not TUNE:
        return base.fit(X_tr, y_tr), None
    search = RandomizedSearchCV(base, PARAM_DIST, n_iter=TUNE_ITER,
                                cv=inner_temporal_split(g_tr),
                                scoring="f1_macro", random_state=seed,
                                n_jobs=1)   # el bosque ya usa todos los núcleos
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_params_


def full_proba(clf, X, n_classes):
    """predict_proba solo trae columnas de las clases vistas en entrenamiento.
    Esta función las acomoda en una matriz con TODAS las clases (las faltantes = 0)."""
    P = np.zeros((len(X), n_classes))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def calibrate_threshold(X_tr, yb_tr, g_tr, target_fpr=TARGET_FPR, val_frac=0.25):
    """Umbral de decisión para el bosque binario (maligno vs. benigno).

    Por defecto un Random Forest marca "maligno" si >= 50 % de los árboles votan
    maligno (umbral 0.5). Con una familia que el modelo NUNCA vio, los árboles
    dudan y el voto se reparte (p. ej. 15-30 %): nunca llega a mayoría aunque
    quede muy por encima de los benignos (que reciben ~0 %).

    Aquí el umbral se elige con datos de ENTRENAMIENTO, sin tocar la prueba:
      1. Las ventanas de entrenamiento se cortan otra vez en el tiempo:
         inicio (75 %) → entrena un bosque auxiliar; final (25 %) → validación.
      2. Se calculan los scores de las ventanas BENIGNAS de validación.
      3. El umbral es el percentil (1 - target_fpr) de esos scores: con
         target_fpr = 0.01, solo el 1 % de las ventanas benignas lo supera.
    Devuelve el umbral (float).
    """
    fit_idx, val_idx = inner_temporal_split(g_tr, val_frac)[0]
    clf_v, _ = build_model(X_tr[fit_idx], yb_tr[fit_idx], g_tr[fit_idx])
    s_val = full_proba(clf_v, X_tr[val_idx], 2)[:, 1]
    s_ben = s_val[yb_tr[val_idx] == 0]
    return float(np.quantile(s_ben, 1 - target_fpr))


def plot_confusion(cm, classes, title, fname):
    """Matriz de confusión: filas = clase real, columnas = clase predicha."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        if tick.get_text() in MALICIOUS:
            tick.set_color("red")
            tick.set_fontweight("bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted class (Random Forest)")
    ax.set_ylabel("True class")
    ax.set_title(title)
    savefig(fname)


def plot_importances(imp_df, title, fname, top=15):
    """Barras horizontales con las features más importantes."""
    d = imp_df.head(top).iloc[::-1]   # al revés para que la más importante quede arriba
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.5))
    ax.barh(d["feature"], d["importance"], color="#4c72b0")
    ax.set_xlabel("Mean decrease in impurity (feature_importances_)")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="x")
    savefig(fname)


# ---------------------------------------------------------------------
# Random Forest: entrenamiento y evaluación
# ---------------------------------------------------------------------
def rf_eval(df, name, prefix, md):
    """Pipeline de Random Forest para un conjunto de features (time / frequency / hybrid).

    Pasos (los del artículo, adaptados a series de tiempo):
      1. X = features, y = clase. Sin escalar y sin PCA: un árbol solo compara
         "¿feature > umbral?", así que la escala no cambia nada.
      2. Split temporal 80/20 dentro de cada corrida.
      3. Entrenar con el 80 %, predecir el 20 % y medir train vs. test (sobreajuste).
      4. Métricas multiclase y maligno vs. benigno.
      5. Matriz de confusión y reporte por clase (sobre el 20 % de prueba).
      6. ROC AUC con la probabilidad de ser maligno que da el bosque.
      7. Importancia de features (feature_importances_, como en el artículo).
      8. Extra: ¿detecta un ransomware que NUNCA vio? (deja fuera una familia).
    """
    df = df.reset_index(drop=True)
    feats = feature_columns(df)
    X = df[feats].to_numpy()
    classes = [c for c in CLASSES if c in set(df["class"])]
    class_to_num = {c: i for i, c in enumerate(classes)}
    y = df["class"].map(class_to_num).to_numpy()
    groups = df["pid"].to_numpy()
    n = len(classes)

    mal_present = [c for c in MALICIOUS if c in class_to_num]
    mal_nums = [class_to_num[c] for c in mal_present]
    y_bin = np.isin(y, mal_nums).astype(int)   # 1 = ransomware, 0 = benigno

    print(f"\n=== {name} ===")
    md.append(f"## Features: {name}\n")
    md.append(f"- Features de entrada: {len(feats)} (sin escalar y sin PCA)")

    # --- Paso 2: split temporal ---
    tr, te = temporal_split(df, TEST_FRAC)
    pct = int(round((1 - TEST_FRAC) * 100))
    print(f"Split temporal {pct}/{100 - pct} dentro de cada corrida: "
          f"{len(tr)} ventanas de entrenamiento, {len(te)} de prueba")
    md.append(f"- Validación: split temporal {pct}/{100 - pct} dentro de cada corrida "
              f"(primer {pct} % de las ventanas entrena, último {100 - pct} % prueba): "
              f"{len(tr)} ventanas de entrenamiento y {len(te)} de prueba")
    md.append("- Entrenamiento y prueba salen de la misma corrida; no mide "
              "generalización a corridas nuevas")
    if not check_split(tr, te, y, classes):
        md.append("- **ERROR: alguna clase quedó sin datos de entrenamiento**")
    md.append(f"- Búsqueda de hiperparámetros: {'sí (RandomizedSearchCV)' if TUNE else 'no (RF_PARAMS fijos)'}\n")

    # --- Pasos 3 y 4: entrenar, predecir y medir ---
    clf, best_params = build_model(X[tr], y[tr], groups[tr])
    y_pred = clf.predict(X[te])
    proba = full_proba(clf, X[te], n)
    y_te, yb_te = y[te], y_bin[te]
    yb_pred = np.isin(y_pred, mal_nums).astype(int)
    # Probabilidad de "maligno" = suma de las probabilidades de todas las clases malignas.
    # Así, confundir cryptsky con ransomware NO cuenta como error de detección.
    score_mal = proba[:, mal_nums].sum(axis=1)
    metrics = {
        # train_acc vs test_acc: si train ≈ 1.0 y test mucho menor → sobreajuste,
        # igual que la comparación clf.score(training) vs clf.score(testing) del artículo.
        "train_acc": clf.score(X[tr], y[tr]),
        "test_acc": accuracy_score(y_te, y_pred),
        "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "mal_precision": precision_score(yb_te, yb_pred, zero_division=0),
        "mal_recall": recall_score(yb_te, yb_pred, zero_division=0),
        "mal_f1": f1_score(yb_te, yb_pred, zero_division=0),
    }
    pd.DataFrame([metrics]).to_csv(OUT / f"{prefix}_rf_metrics.csv", index=False)
    print(pd.Series(metrics).round(3).to_string())
    md.append("### Random Forest (split temporal)\n")
    md.append(md_table(["Métrica", "Valor"], [[k, v] for k, v in metrics.items()]))
    md.append("")

    result = {"features": name, "n_feat": len(feats)}
    for m in ["train_acc", "test_acc", "f1_macro", "mal_precision", "mal_recall", "mal_f1"]:
        result[m] = f"{metrics[m]:.3f}"

    if best_params:
        md.append("### Mejores hiperparámetros (RandomizedSearchCV)\n")
        md.append(f"`{best_params}`\n")
        pd.DataFrame([best_params]).to_csv(OUT / f"{prefix}_rf_best_params.csv", index=False)

    # --- Paso 5: matriz de confusión y reporte por clase (20 % de prueba) ---
    cm = confusion_matrix(y_te, y_pred, labels=range(n))
    plot_confusion(cm, classes, f"Confusion matrix — RF, {name} (test {100 - pct} %)",
                   f"{prefix}_rf_confusionmatrix.png")
    report = classification_report(y_te, y_pred, labels=range(n), target_names=classes,
                                   zero_division=0, digits=3)
    print(report)
    md.append(f"### Reporte por clase (último {100 - pct} % de cada corrida)\n")
    md.append("```\n" + report + "```\n")

    # --- Paso 6: ROC AUC (sobre el conjunto de prueba) ---
    if not mal_present:
        md.append("_No hay clases malignas en los datos; se omite el ROC AUC._\n")
        return result

    benign = yb_te == 0
    auc_rows = []
    for cname in mal_present:
        num = class_to_num[cname]
        m = benign | (y_te == num)   # solo esta familia vs. benignos
        auc = roc_auc_score((y_te[m] == num).astype(int), proba[m, num])
        auc_rows.append([f"{cname} vs benignos", auc])
        result[f"AUC {cname}"] = f"{auc:.3f}"
    auc_all = roc_auc_score(yb_te, score_mal)
    auc_rows.append([f"malignos ({' + '.join(mal_present)}) vs benignos", auc_all])
    result["AUC malignos"] = f"{auc_all:.3f}"
    for label, auc in auc_rows:
        print(f"ROC AUC {label}: {auc:.3f}")
    md.append("### ROC AUC (probabilidad de maligno del Random Forest, conjunto de prueba)\n")
    md.append(md_table(["Comparación", "AUC"], auc_rows))
    md.append("")

    # --- Paso 7: importancia de features ---
    imp_df = (pd.DataFrame({"feature": feats, "importance": clf.feature_importances_})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))
    imp_df.to_csv(OUT / f"{prefix}_rf_feature_importance.csv", index=False)
    plot_importances(imp_df, f"Feature importance — RF, {name}",
                     f"{prefix}_rf_feature_importance.png")
    md.append("### Features más importantes (top 10)\n")
    md.append(md_table(["Feature", "Importancia"], imp_df.head(10).values.tolist()))
    md.append("")

    # --- Paso 8: ransomware nunca visto (leave-one-family-out) ---
    # Pregunta realista: si aparece un ransomware NUEVO, ¿lo detecta?
    # Se entrena un bosque binario (maligno vs. benigno) SIN ninguna ventana de
    # una familia (usando el 80 % de entrenamiento de los demás), y se prueba sobre
    # el 20 % final de esa familia + el 20 % final de los benignos.
    #   detección = % de ventanas de la familia nueva marcadas como malignas
    #   FPR       = % de ventanas benignas marcadas como malignas (falsas alarmas)
    # Se reporta con dos umbrales: 0.5 (mayoría de árboles, el de sklearn) y uno
    # calibrado con calibrate_threshold() para un FPR objetivo de TARGET_FPR.
    if len(mal_present) >= 2:
        lofo_rows = []
        for cname in mal_present:
            num = class_to_num[cname]
            tr_f = tr[y[tr] != num]                        # el modelo NUNCA ve esta familia
            te_f = te[(y_bin[te] == 0) | (y[te] == num)]   # prueba: benignos + familia nueva
            clf_f, _ = build_model(X[tr_f], y_bin[tr_f], groups[tr_f])
            thr = calibrate_threshold(X[tr_f], y_bin[tr_f], groups[tr_f])
            score = full_proba(clf_f, X[te_f], 2)[:, 1]
            yt = (y[te_f] == num).astype(int)
            pred05 = (score >= 0.5).astype(int)
            pred_cal = (score > thr).astype(int)
            det05 = recall_score(yt, pred05, zero_division=0)
            fpr05 = pred05[yt == 0].mean()
            det_cal = recall_score(yt, pred_cal, zero_division=0)
            fpr_cal = pred_cal[yt == 0].mean()
            auc = roc_auc_score(yt, score)
            lofo_rows.append([cname, det05, fpr05, thr, det_cal, fpr_cal, auc])
            result[f"det. {cname} no visto"] = f"{det05:.3f}"
            result[f"det. {cname} no visto (cal)"] = f"{det_cal:.3f}"
            print(f"{cname} nunca visto → umbral 0.5: detección {det05:.3f}, FPR {fpr05:.3f} | "
                  f"umbral {thr:.3f}: detección {det_cal:.3f}, FPR {fpr_cal:.3f} | AUC {auc:.3f}")
        md.append("### Generalización: ransomware nunca visto en entrenamiento\n")
        md.append("Bosque binario (maligno vs. benigno) entrenado SIN la familia indicada. "
                  f"El umbral calibrado se elige en validación (final del entrenamiento) para "
                  f"un FPR objetivo de {TARGET_FPR:.0%}; el FPR de la tabla es el medido en prueba.\n")
        md.append(md_table(["Familia excluida", "Detección (0.5)", "FPR (0.5)",
                            "Umbral calibrado", "Detección (cal)", "FPR (cal)", "AUC"],
                           lofo_rows))
        md.append("")

    return result


# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------
def run_window(raw: pd.DataFrame, window_s: int):
    """Corre el pipeline completo para UN tamaño de ventana."""
    set_window(window_s)
    print(f"\n{'#' * 60}\n# Ventana de {WINDOW_S} s ({WINDOW_N} muestras) → {OUT}\n{'#' * 60}")

    raw.to_csv(OUT / "pcm_telemetry.csv", index=False)

    print("Extracting window features...")
    time_df, freq_df, hybrid_df = extract_features(raw)
    time_df.to_csv(OUT / "features_time_domain.csv", index=False)
    freq_df.to_csv(OUT / "features_frequency_domain.csv", index=False)
    hybrid_df.to_csv(OUT / "features_hybrid.csv", index=False)

    print("Generating explanatory charts...")
    plot_raw_examples(raw)
    plot_window_example(raw)
    plot_same_mean_different_structure()
    plot_fft_examples(raw)
    pca_plot(time_df, "PCA (visualization only) — time-domain features",
             "07_pca_time_domain.png")
    pair_md = plot_metric_pairs(time_df)

    print("Random Forest...")
    counts = time_df["class"].value_counts()
    md = [
        "# Estadísticas — PCM, Random Forest\n",
        f"- Generado: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Datos: `{DATA_DIR}`",
        f"- Ventana: {WINDOW_S} s ({WINDOW_N} muestras a {FS:g} Hz)",
        f"- Métricas: {', '.join(METRICS)}",
        f"- Clases malignas: {', '.join(MALICIOUS)}",
        f"- FPR objetivo del umbral calibrado: {TARGET_FPR:.0%}\n",
        "### Ventanas por clase\n",
        md_table(["Clase", "Tipo", "Corridas", "Ventanas"],
                 [[c, "maligna" if c in MALICIOUS else "benigna",
                   raw.loc[raw["class"] == c, "pid"].nunique(), int(counts.get(c, 0))]
                  for c in CLASSES if c in counts.index]),
        "",
    ]

    results = [
        rf_eval(time_df, "time", "08_time", md),
        rf_eval(freq_df, "frequency", "09_freq", md),
        rf_eval(hybrid_df, "hybrid", "10_hybrid", md),
    ]

    cols = list(results[-1].keys())
    md.append("## Comparación entre familias de features\n")
    md.append(md_table(cols, [[r.get(c, "") for c in cols] for r in results]))
    md.append("")
    md.extend(pair_md)

    (OUT / "stats.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Estadísticas guardadas en {OUT / 'stats.md'}")
    return results


def write_run_summary(raw: pd.DataFrame):
    """Diagnóstico: promedio de cada métrica por CORRIDA (runs_summary.md / .csv).

    Si las dos corridas de una misma aplicación tienen promedios muy distintos
    (p. ej. CPI 0.8 en una y 1.6 en otra), el modelo no puede aprender la clase
    a partir de una sola corrida. Esta tabla sirve para detectar eso.
    La columna "dif_vs_clase_%" es cuánto se aleja el CPI de esa corrida del
    promedio de su clase.
    """
    s = (raw.groupby(["class", "pid", "run_file"], sort=False)[METRICS]
         .mean().reset_index())
    s.insert(3, "muestras", raw.groupby("pid", sort=False).size().to_numpy())
    if s.groupby("class")["pid"].nunique().max() > 1:
        class_cpi = s.groupby("class")["CPI"].transform("mean")
        s["dif_vs_clase_%"] = (s["CPI"] - class_cpi) / class_cpi * 100
    s.to_csv(BASE_OUT / "runs_summary.csv", index=False)
    md = [
        "# Promedio de cada métrica por corrida\n",
        "Si las corridas de una misma clase difieren mucho entre sí, el modelo no generaliza.\n",
        md_table(list(s.columns), s.values.tolist()),
        "",
    ]
    (BASE_OUT / "runs_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\nPromedio por corrida (guardado en runs_summary.md):")
    print(s.drop(columns="run_file").round(3).to_string(index=False))


def select_run(raw: pd.DataFrame, k: int) -> pd.DataFrame:
    """Se queda solo con la k-ésima corrida de cada workload (k = 1 es la primera,
    en el orden alfabético de los CSV). Si un workload tiene menos de k corridas,
    se usa su última corrida y se avisa."""
    keep = []
    for label, g in raw.groupby("class", sort=False):
        pids = list(dict.fromkeys(g["pid"]))
        if len(pids) < k:
            print(f"[aviso] '{label}' solo tiene {len(pids)} corrida(s); se usa la última")
        keep.append(pids[min(k, len(pids)) - 1])
    out = raw[raw["pid"].isin(keep)]
    files = out.groupby("class", sort=False)["run_file"].first()
    print(f"Se usa la corrida {k} de cada workload:")
    for label, f in files.items():
        print(f"  {label:<11} {f}")
    return out


def main():
    BASE_OUT.mkdir(exist_ok=True)
    print(f"Split temporal: entrenamiento = primer {1 - TEST_FRAC:.0%} de cada corrida, "
          f"prueba = último {TEST_FRAC:.0%}; corrida(s) usada(s): {RUN_CHOICE}  →  salida en {BASE_OUT}/")

    print("Loading PCM telemetry...")
    raw = load_pcm_dataset(min_samples=int(FS * max(WINDOW_SIZES_S)))
    if RUN_CHOICE != "all":
        raw = select_run(raw, int(RUN_CHOICE))
    write_run_summary(raw)

    all_results = []
    for window_s in WINDOW_SIZES_S:
        for r in run_window(raw, window_s):
            all_results.append({"window_s": window_s, **r})

    cols = ["window_s"] + [c for c in all_results[-1] if c != "window_s"]
    md = [
        "# Comparación entre tamaños de ventana — Random Forest\n",
        f"- Generado: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Ventanas evaluadas: {', '.join(f'{w} s' for w in WINDOW_SIZES_S)}",
        "- El detalle de cada ventana está en `window_XXs/stats.md`.\n",
        md_table(cols, [[r.get(c, "") for c in cols] for r in all_results]),
        "",
    ]
    (BASE_OUT / "stats_all_windows.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nDone. Outputs written to: {BASE_OUT.resolve()}")


if __name__ == "__main__":
    main()
