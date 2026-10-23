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
   ransomware).
2. CALCULA 4 métricas por muestra: CPI, IPC, LLC_MPKI y L2_MPKI.
3. CORTA cada corrida en VENTANAS de 1, 2, 5, 10, 15 y 20 segundos.
4. Para cada ventana EXTRAE features (tiempo, frecuencia FFT e híbrido).
5. ENTRENA un Random Forest y lo evalúa con validación cruzada AGRUPADA POR
   CORRIDA: las ventanas de una misma corrida nunca están a la vez en
   entrenamiento y en prueba (si no, el modelo "se copia" y los números salen
   inflados).
6. Mide precision, recall, F1 (por clase y maligno vs. benigno), ROC AUC,
   sobreajuste (train vs. test), importancia de features y una prueba de
   generalización: ¿detecta un ransomware que NUNCA vio al entrenar?
   Todo se guarda en ./pcm_rf_output/window_XXs/ y la tabla comparativa en
   ./pcm_rf_output/stats_all_windows.md

Dependencias (scikit-learn >= 1.0 por StratifiedGroupKFold):
    pip install numpy pandas matplotlib scikit-learn scipy

Ejecutar:
    python pcm_time_frequency_rf.py
    # o con otra carpeta de datos:
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score, classification_report)


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

METRICS = ["CPI", "IPC", "LLC_MPKI", "L2_MPKI"]

CLASSES = ["7zip", "ffmpeg", "gpg", "openssl", "readwrite", "rsync", "stress",
           "sysbench", "mlc", "cryptsky", "ransomware"]

MALICIOUS = ["cryptsky", "ransomware"]

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

# Cómo se separan entrenamiento y prueba (variable de entorno PCM_SPLIT):
#   "run"    → por corrida completa (StratifiedGroupKFold). Es la evaluación honesta:
#              el modelo se prueba con corridas que nunca vio. Valor por defecto.
#   "window" → por ventanas sueltas (StratifiedKFold), IGNORANDO la corrida.
#              SOLO PARA DIAGNÓSTICO: ventanas de la misma corrida quedan en
#              entrenamiento y en prueba. Si aquí sale ~1.0 y con "run" sale bajo,
#              el modelo distingue corridas que ya vio pero no generaliza a corridas nuevas.
SPLIT_MODE = os.environ.get("PCM_SPLIT", "run")
if SPLIT_MODE not in ("run", "window"):
    raise ValueError("PCM_SPLIT debe ser 'run' o 'window'")

# Carpeta de salida (distinta a la de K-means; y otra distinta para el modo diagnóstico).
BASE_OUT = Path("pcm_rf_output" if SPLIT_MODE == "run" else "pcm_rf_output_window_split")
OUT = BASE_OUT

# ---------------------------------------------------------------------
# Configuración del Random Forest
# ---------------------------------------------------------------------
# Número de pliegues (folds) de la validación cruzada, igual que cv=5 en el artículo.
# Si alguna clase tiene menos corridas que esto, se reduce automáticamente.
N_SPLITS = 5

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
# La búsqueda se hace SOLO con los datos de entrenamiento de cada pliegue
# (validación cruzada anidada), para no "espiar" el conjunto de prueba.
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


def make_group_cv(df, max_splits, seed=42):
    """Validación cruzada estratificada AGRUPADA POR CORRIDA.

    ¿Por qué no usar train_test_split como en el artículo?
    En el artículo cada fila (un café) es independiente. Aquí NO: dos ventanas
    seguidas de la misma corrida son casi idénticas. Si una cae en entrenamiento
    y su "gemela" en prueba, el modelo acierta por memoria y el F1 sale inflado.
    StratifiedGroupKFold manda corridas COMPLETAS a prueba (group = pid) y trata
    de que cada pliegue tenga todas las clases (stratified).

    El número de pliegues no puede ser mayor que las corridas de la clase con
    menos corridas: con 3 corridas de cryptsky, máximo 3 pliegues.
    """
    runs = df.groupby("class")["pid"].nunique()
    n = int(min(max_splits, runs.min()))
    if n < 2:
        return None, runs
    if n < max_splits:
        print(f"[aviso] la clase con menos corridas tiene {runs.min()}; "
              f"se usan {n} pliegues en lugar de {max_splits}")
    return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed), runs


def build_model(X_tr, y_tr, g_tr, seed):
    """Entrena el Random Forest con los datos de entrenamiento de un pliegue.

    TUNE=False → parámetros fijos (RF_PARAMS), equivalente al
                 `clf = RandomForestClassifier(); clf.fit(...)` del artículo.
    TUNE=True  → RandomizedSearchCV como en el artículo, pero su validación
                 interna también agrupa por corrida y solo ve datos de entrenamiento.
    Devuelve (modelo_entrenado, mejores_parámetros o None).
    """
    base = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
    if not TUNE:
        return base.fit(X_tr, y_tr), None

    runs_min = pd.Series(g_tr).groupby(y_tr).nunique().min()
    if runs_min < 2:
        print("[aviso] muy pocas corridas para la búsqueda interna; se usan RF_PARAMS")
        return base.fit(X_tr, y_tr), None
    inner = StratifiedGroupKFold(n_splits=int(min(3, runs_min)), shuffle=True,
                                 random_state=seed)
    search = RandomizedSearchCV(base, PARAM_DIST, n_iter=TUNE_ITER, cv=inner,
                                scoring="f1_macro", random_state=seed,
                                n_jobs=1)   # el bosque ya usa todos los núcleos
    search.fit(X_tr, y_tr, groups=g_tr)
    return search.best_estimator_, search.best_params_


def full_proba(clf, X, n_classes):
    """predict_proba solo trae columnas de las clases vistas en entrenamiento.
    Esta función las acomoda en una matriz con TODAS las clases (las faltantes = 0)."""
    P = np.zeros((len(X), n_classes))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


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
    """Barras horizontales con las features más importantes (media ± desv. entre pliegues)."""
    d = imp_df.head(top).iloc[::-1]   # al revés para que la más importante quede arriba
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#4c72b0")
    ax.set_xlabel("Mean decrease in impurity (feature_importances_)")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="x")
    savefig(fname)


# ---------------------------------------------------------------------
# Random Forest: entrenamiento y evaluación
# ---------------------------------------------------------------------
def rf_eval(df, name, prefix, md):
    """Pipeline de Random Forest para un conjunto de features (time / frequency / hybrid).

    Pasos (los del artículo, adaptados):
      1. X = features, y = clase. Sin escalar y sin PCA: un árbol solo compara
         "¿feature > umbral?", así que la escala no cambia nada.
      2. Validación cruzada agrupada por corrida (en vez de un solo split 75/25).
      3. En cada pliegue: entrenar, predecir, medir train vs. test (sobreajuste).
      4. Métricas promedio ± desviación entre pliegues (multiclase y maligno vs. benigno).
      5. Predicciones "out-of-fold": cada ventana predicha por el modelo que NO
         vio su corrida → matriz de confusión y reporte por clase.
      6. ROC AUC con la probabilidad de ser maligno que da el bosque.
      7. Importancia de features (feature_importances_, como en el artículo).
      8. Extra: ¿detecta un ransomware que NUNCA vio? (deja fuera una familia).
    """
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

    # --- Paso 2: pliegues agrupados por corrida ---
    cv, runs = make_group_cv(df, N_SPLITS)
    if cv is None:
        msg = (f"_Alguna clase tiene solo {runs.min()} corrida(s); se necesitan al menos 2 "
               "por clase para separar entrenamiento y prueba por corrida._\n")
        print(msg)
        md.append(msg)
        return {"features": name}
    if SPLIT_MODE == "window":
        # Modo diagnóstico: ventanas sueltas, sin respetar la corrida.
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))
        md.append(f"- Validación: StratifiedKFold, {N_SPLITS} pliegues por VENTANA "
                  "(diagnóstico: ventanas de una misma corrida en train y test)")
    else:
        splits = list(cv.split(X, y, groups))
        md.append(f"- Validación: StratifiedGroupKFold, {cv.n_splits} pliegues agrupados por corrida")
    md.append(f"- Búsqueda de hiperparámetros: {'sí (RandomizedSearchCV anidado)' if TUNE else 'no (RF_PARAMS fijos)'}\n")

    # --- Pasos 3 y 4: entrenar y medir en cada pliegue ---
    y_pred_oof = np.full(len(y), -1)
    proba_oof = np.zeros((len(y), n))
    importances, fold_rows, best_params = [], [], []

    for fold, (tr, te) in enumerate(splits):
        clf, bp = build_model(X[tr], y[tr], groups[tr], seed=fold)
        if bp:
            best_params.append(bp)
        y_pred = clf.predict(X[te])
        proba = full_proba(clf, X[te], n)
        y_pred_oof[te] = y_pred
        proba_oof[te] = proba
        importances.append(clf.feature_importances_)

        yb_te = y_bin[te]
        yb_pred = np.isin(y_pred, mal_nums).astype(int)
        # Probabilidad de "maligno" = suma de las probabilidades de todas las clases malignas.
        # Así, confundir cryptsky con ransomware NO cuenta como error de detección.
        score_mal = proba[:, mal_nums].sum(axis=1)
        fold_rows.append({
            "fold": fold,
            "n_test_runs": len(np.unique(groups[te])),
            # train_acc vs test_acc: si train ≈ 1.0 y test mucho menor → sobreajuste,
            # igual que la comparación clf.score(training) vs clf.score(testing) del artículo.
            "train_acc": clf.score(X[tr], y[tr]),
            "test_acc": accuracy_score(y[te], y_pred),
            "precision_macro": precision_score(y[te], y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y[te], y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y[te], y_pred, average="macro", zero_division=0),
            "mal_precision": precision_score(yb_te, yb_pred, zero_division=0),
            "mal_recall": recall_score(yb_te, yb_pred, zero_division=0),
            "mal_f1": f1_score(yb_te, yb_pred, zero_division=0),
            "AUC_mal": roc_auc_score(yb_te, score_mal) if len(np.unique(yb_te)) == 2 else np.nan,
        })

    metrics = pd.DataFrame(fold_rows)
    metrics.to_csv(OUT / f"{prefix}_rf_fold_metrics.csv", index=False)
    summary = metrics.drop(columns=["fold", "n_test_runs"]).agg(["mean", "std"]).T
    print(summary.round(3))
    md.append(f"### Random Forest ({cv.n_splits} pliegues por corrida)\n")
    md.append(md_table(["Métrica", "Media", "Desv. est."],
                       [[m, r["mean"], r["std"]] for m, r in summary.iterrows()]))
    md.append("")

    result = {"features": name, "n_feat": len(feats)}
    for m in ["train_acc", "f1_macro", "mal_precision", "mal_recall", "mal_f1"]:
        result[m] = f"{summary.loc[m, 'mean']:.3f} ± {summary.loc[m, 'std']:.3f}"

    if best_params:
        md.append("### Mejores hiperparámetros por pliegue (RandomizedSearchCV)\n")
        for i, bp in enumerate(best_params):
            md.append(f"- Pliegue {i}: `{bp}`")
        md.append("")
        pd.DataFrame(best_params).to_csv(OUT / f"{prefix}_rf_best_params.csv", index=False)

    # --- Paso 5: matriz de confusión y reporte por clase (out-of-fold) ---
    cm = confusion_matrix(y, y_pred_oof, labels=range(n))
    plot_confusion(cm, classes, f"Confusion matrix — RF, {name} (out-of-fold)",
                   f"{prefix}_rf_confusionmatrix.png")
    report = classification_report(y, y_pred_oof, labels=range(n), target_names=classes,
                                   zero_division=0, digits=3)
    print(report)
    md.append("### Reporte por clase (predicciones out-of-fold)\n")
    md.append("```\n" + report + "```\n")

    # --- Paso 6: ROC AUC (out-of-fold) ---
    if not mal_present:
        md.append("_No hay clases malignas en los datos; se omite el ROC AUC._\n")
        return result

    benign = y_bin == 0
    auc_rows = []
    for cname in mal_present:
        num = class_to_num[cname]
        m = benign | (y == num)   # solo esta familia vs. benignos
        auc = roc_auc_score((y[m] == num).astype(int), proba_oof[m, num])
        auc_rows.append([f"{cname} vs benignos", auc])
        result[f"AUC {cname}"] = f"{auc:.3f}"
    auc_all = roc_auc_score(y_bin, proba_oof[:, mal_nums].sum(axis=1))
    auc_rows.append([f"malignos ({' + '.join(mal_present)}) vs benignos", auc_all])
    result["AUC malignos"] = f"{auc_all:.3f}"
    for label, auc in auc_rows:
        print(f"ROC AUC {label}: {auc:.3f}")
    md.append("### ROC AUC (probabilidad de maligno del Random Forest, out-of-fold)\n")
    md.append(md_table(["Comparación", "AUC"], auc_rows))
    md.append("")

    # --- Paso 7: importancia de features ---
    imp = np.array(importances)
    imp_df = (pd.DataFrame({"feature": feats,
                            "importance_mean": imp.mean(axis=0),
                            "importance_std": imp.std(axis=0)})
              .sort_values("importance_mean", ascending=False)
              .reset_index(drop=True))
    imp_df.to_csv(OUT / f"{prefix}_rf_feature_importance.csv", index=False)
    plot_importances(imp_df, f"Feature importance — RF, {name}",
                     f"{prefix}_rf_feature_importance.png")
    md.append("### Features más importantes (top 10)\n")
    md.append(md_table(["Feature", "Importancia", "Desv. est."],
                       imp_df.head(10).values.tolist()))
    md.append("")

    # --- Paso 8: ransomware nunca visto (leave-one-family-out) ---
    # Pregunta realista: si aparece un ransomware NUEVO, ¿lo detecta?
    # Se entrena un bosque binario (maligno vs. benigno) SIN ninguna ventana de
    # una familia, y se prueba sobre esa familia + corridas benignas no vistas.
    #   detección = % de ventanas de la familia nueva marcadas como malignas
    #   FPR       = % de ventanas benignas marcadas como malignas (falsas alarmas)
    if len(mal_present) >= 2:
        lofo_rows = []
        for cname in mal_present:
            num = class_to_num[cname]
            score = np.full(len(y), np.nan)
            for fold, (tr, te) in enumerate(splits):
                tr = tr[y[tr] != num]                        # el modelo NUNCA ve esta familia
                te = te[(y_bin[te] == 0) | (y[te] == num)]   # prueba: benignos + familia nueva
                clf, _ = build_model(X[tr], y_bin[tr], groups[tr], seed=fold)
                score[te] = full_proba(clf, X[te], 2)[:, 1]
            m = ~np.isnan(score)
            yt = (y[m] == num).astype(int)
            pred = (score[m] >= 0.5).astype(int)
            det = recall_score(yt, pred, zero_division=0)
            fpr = pred[yt == 0].mean()
            auc = roc_auc_score(yt, score[m])
            lofo_rows.append([cname, det, fpr, auc])
            result[f"det. {cname} no visto"] = f"{det:.3f}"
            print(f"{cname} nunca visto → detección {det:.3f}, FPR {fpr:.3f}, AUC {auc:.3f}")
        md.append("### Generalización: ransomware nunca visto en entrenamiento\n")
        md.append("Bosque binario (maligno vs. benigno) entrenado SIN la familia indicada.\n")
        md.append(md_table(["Familia excluida", "Detección (recall)", "FPR benignos", "AUC"],
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

    print("Random Forest...")
    counts = time_df["class"].value_counts()
    md = [
        "# Estadísticas — PCM, Random Forest\n",
        f"- Generado: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Datos: `{DATA_DIR}`",
        f"- Ventana: {WINDOW_S} s ({WINDOW_N} muestras a {FS:g} Hz)",
        f"- Métricas: {', '.join(METRICS)}",
        f"- Clases malignas: {', '.join(MALICIOUS)}\n",
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


def main():
    BASE_OUT.mkdir(exist_ok=True)
    print(f"Modo de separación train/test: {SPLIT_MODE}  →  salida en {BASE_OUT}/")

    print("Loading PCM telemetry...")
    raw = load_pcm_dataset(min_samples=int(FS * max(WINDOW_SIZES_S)))
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
