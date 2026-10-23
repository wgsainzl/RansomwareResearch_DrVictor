#!/usr/bin/env python3
"""
Caracterización microarquitectónica de cargas de trabajo con PCM
================================================================

¿Qué hace este script, en pocas palabras?
-----------------------------------------
1. LEE los CSV que genera Intel PCM (contadores de hardware del CPU) para
   varias aplicaciones: 7zip, ffmpeg, gpg, ..., y los ransomware
   "cryptsky" y "ransomware".
2. CALCULA 4 métricas por muestra: CPI, IPC, LLC_MPKI y L2_MPKI.
3. CORTA cada corrida en VENTANAS de 1, 2, 5, 10, 15 y 20 segundos
   (el análisis completo se repite para cada tamaño, en su propia carpeta).
4. Para cada ventana EXTRAE "features" (números que la resumen):
     - Dominio del tiempo:      promedio y variación.
     - Dominio de la frecuencia: qué tan periódica/ruidosa es la señal (FFT).
     - Híbrido:                 ambas cosas juntas.
5. Reduce dimensiones con PCA y agrupa con K-means (sin usar etiquetas)
   para ver si las aplicaciones se separan "solas".
6. Mide qué tan bien salió (precision, recall, F1, ARI, silhouette, AUC),
   tanto por clase como en modo binario maligno vs. benigno,
   y guarda gráficas, CSV y un resumen stats.md en ./pcm_demo_output/window_XXs/,
   más una tabla comparativa de todas las ventanas en
   ./pcm_demo_output/stats_all_windows.md

Dependencias:
    pip install numpy pandas matplotlib scikit-learn scipy

Ejecutar:
    python pcm_time_frequency_demo.py
"""

from pathlib import Path
from datetime import datetime
import numpy as np                  # cálculo numérico (arreglos, FFT)
import pandas as pd                 # tablas (DataFrames) y lectura de CSV
import matplotlib.pyplot as plt     # gráficas

from sklearn.preprocessing import StandardScaler   # normaliza cada columna (media 0, desv. 1)
from sklearn.decomposition import PCA              # Análisis de Componentes Principales
from sklearn.cluster import KMeans                 # agrupamiento no supervisado
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             adjusted_rand_score, silhouette_score, roc_auc_score)
from scipy.optimize import linear_sum_assignment   # algoritmo húngaro (emparejar clusters ↔ clases)


# Generador de números aleatorios con semilla fija → resultados reproducibles.
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
# Carpeta con los resultados. Se espera una subcarpeta por clase, por ejemplo:
#   ~/Desktop/Research/Resultados_ALL/7zip/xxx_pcm_1.csv
#   ~/Desktop/Research/Resultados_ALL/cryptsky/xxx_pcm_1.csv
DATA_DIR = Path.home() / "Desktop" / "Research" / "Resultados_ALL"

FS = 10.0                  # Frecuencia de muestreo: 10 muestras por segundo (PCM cada 0.1 s)

# Tamaños de ventana (en segundos) a evaluar. El pipeline completo se corre
# una vez por cada tamaño y deja sus resultados en su propia carpeta:
#   pcm_demo_output/window_01s/, window_02s/, ..., window_20s/
WINDOW_SIZES_S = [1, 2, 5, 10, 15, 20]

# Ventana "actual". main() las cambia en cada iteración con set_window();
# todas las funciones leen estos valores al momento de ejecutarse.
WINDOW_S = 20              # Duración de cada ventana en segundos
WINDOW_N = int(FS * WINDOW_S)   # Muestras por ventana: 10 × 20 = 200

# Métricas que se analizan para cada muestra:
#   IPC      = instrucciones por ciclo (más alto = el CPU trabaja más "fluido")
#   CPI      = ciclos por instrucción = 1 / IPC (más alto = más esperas)
#   LLC_MPKI = fallos de la caché L3 (último nivel) por cada 1000 instrucciones
#   L2_MPKI  = fallos de la caché L2 por cada 1000 instrucciones
METRICS = ["CPI", "IPC", "LLC_MPKI", "L2_MPKI"]

# Clases (aplicaciones) a cargar. Las que están en MALICIOUS son ransomware;
# el resto son cargas benignas que sirven de comparación.
CLASSES = ["7zip", "ffmpeg", "gpg", "openssl", "readwrite", "rsync", "stress",
           "sysbench", "mlc", "cryptsky", "ransomware"]

# Clases malignas: se dibujan con "X" en el PCA y se usan en las métricas
# de detección (maligno vs. benigno) y en el ROC AUC.
# Para agregar otro ransomware en el futuro, basta con añadirlo aquí y en CLASSES.
MALICIOUS = ["cryptsky", "ransomware"]

# Colores fijos por clase. La paleta por defecto de matplotlib solo tiene 10
# colores y con 11 clases se repite (ransomware quedaba igual que 7zip).
# Benignos: paleta tab10 SIN el rojo. Malignos: rojo, negro y, si se agregan más, otros oscuros.
_BENIGN_PALETTE = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 3]
_MAL_PALETTE = ["#d62728", "#000000", "#8b0000", "#4b0082"]
COLORS = {}
for _c in CLASSES:
    if _c in MALICIOUS:
        COLORS[_c] = _MAL_PALETTE[MALICIOUS.index(_c) % len(_MAL_PALETTE)]
    else:
        _i = len([k for k in COLORS if k not in MALICIOUS])
        COLORS[_c] = _BENIGN_PALETTE[_i % len(_BENIGN_PALETTE)]


# Carpeta raíz de salidas. Dentro se crea una subcarpeta por tamaño de ventana;
# OUT apunta a la subcarpeta de la ventana que se está procesando.
BASE_OUT = Path("pcm_demo_output")
OUT = BASE_OUT


def set_window(window_s: int):
    """Fija la ventana actual (WINDOW_S, WINDOW_N) y la carpeta de salida OUT.
    Ejemplo: window_s=5 → 50 muestras por ventana y salida en pcm_demo_output/window_05s/"""
    global WINDOW_S, WINDOW_N, OUT
    WINDOW_S = window_s
    WINDOW_N = int(FS * window_s)
    OUT = BASE_OUT / f"window_{window_s:02d}s"
    OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Carga de la telemetría de PCM
# ---------------------------------------------------------------------
def load_pcm_csv(path: Path, label: str, run_id: int) -> pd.DataFrame:
    """Lee un CSV de pcm (2 filas de encabezado) y devuelve el formato largo.

    PCM escribe dos filas de encabezado: la primera es el "grupo"
    (System, Socket 0, Core 0, ...) y la segunda el nombre del contador
    (IPC, INST, L3MISS, ...). Aquí solo usamos el grupo "System", es decir,
    los totales de toda la máquina.

    Devuelve una tabla con una fila por muestra y las columnas:
        time_s, pid, class, IPC, CPI, LLC_MPKI, L2_MPKI, run_file
    """
    # header=[0, 1] → pandas lee las dos filas de encabezado como columnas de 2 niveles.
    df = pd.read_csv(path, header=[0, 1])
    sysd = df["System"]   # nos quedamos solo con las columnas del grupo "System"

    # Pequeña ayuda: toma una columna y la convierte a número
    # (si hay texto raro, lo vuelve NaN en lugar de fallar).
    def col(name):
        return pd.to_numeric(sysd[name], errors="coerce")

    ipc = col("IPC")
    out = pd.DataFrame({
        "IPC": ipc,
        # CPI es el inverso del IPC. Si IPC = 0 se pone NaN para no dividir entre cero.
        "CPI": 1.0 / ipc.where(ipc > 0),
        # MPKI = fallos / instrucciones × 1000 → "fallos por cada mil instrucciones".
        # Normalizar por instrucciones permite comparar programas rápidos y lentos.
        "LLC_MPKI": col("L3MISS") / col("INST") * 1000.0,
        "L2_MPKI": col("L2MISS") / col("INST") * 1000.0,
    })

    # Rellenar huecos (NaN/inf) interpolando para no romper el espaciado temporal.
    # Si borráramos filas, las muestras dejarían de estar separadas 0.1 s
    # y la FFT daría frecuencias incorrectas.
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.interpolate(limit_direction="both")

    # Columnas de identificación:
    #   time_s → tiempo en segundos (muestra 0 = 0 s, muestra 1 = 0.1 s, ...)
    #   pid    → aquí NO es el PID del proceso, sino un número único por corrida/archivo
    #   class  → nombre de la aplicación (la "etiqueta verdadera")
    out.insert(0, "time_s", np.arange(len(out)) / FS)
    out.insert(1, "pid", run_id)          # "pid" = id de corrida
    out.insert(2, "class", label)
    out["run_file"] = path.name           # nombre del CSV de origen, para rastrear
    return out


def load_pcm_dataset(min_samples: int) -> pd.DataFrame:
    """Recorre todas las clases, lee todos sus CSV de PCM y los une en una sola tabla.
    Se omiten las corridas con menos de `min_samples` muestras."""
    frames = []
    run_id = 1000   # los ids de corrida empiezan en 1000 y suben de uno en uno
    for label in CLASSES:
        # Busca archivos cuyo nombre contenga "pcm" dentro de la carpeta de la clase.
        files = sorted((DATA_DIR / label).glob("*pcm*.csv"))
        if not files:
            print(f"[aviso] sin CSV para '{label}' en {DATA_DIR / label}")
            continue
        for f in files:
            d = load_pcm_csv(f, label, run_id)
            # Si la corrida es más corta que la ventana más grande, se omite.
            # Así TODAS las ventanas usan exactamente las mismas corridas
            # y la comparación entre tamaños de ventana es justa.
            if len(d) < min_samples:
                print(f"[aviso] {f.name}: {len(d)} muestras (< {min_samples}), se omite")
                continue
            frames.append(d)
            run_id += 1
    # Une todas las corridas, una debajo de otra.
    return pd.concat(frames, ignore_index=True)


def first_pid(raw: pd.DataFrame, label: str):
    """Devuelve el id de la primera corrida de una clase, o None si la clase no tiene datos.
    Evita que las gráficas truenen si alguna carpeta está vacía."""
    m = raw["class"] == label
    return raw.loc[m, "pid"].iloc[0] if m.any() else None


# ---------------------------------------------------------------------
# Ventaneo y extracción de features
# ---------------------------------------------------------------------
def iter_windows(df: pd.DataFrame):
    """Corta cada corrida en ventanas consecutivas de WINDOW_N muestras.

    Ejemplo: una corrida de 650 muestras da 3 ventanas
    (0–199, 200–399, 400–599); las 50 muestras sobrantes se descartan.
    Las ventanas NO se traslapan.

    Es un generador: entrega (pid, clase, número de ventana, datos) una a la vez.
    """
    for (pid, label), g in df.groupby(["pid", "class"], sort=False):
        g = g.sort_values("time_s").reset_index(drop=True)
        n_windows = len(g) // WINDOW_N   # división entera → solo ventanas completas

        for w in range(n_windows):
            start = w * WINDOW_N
            stop = start + WINDOW_N
            win = g.iloc[start:stop].copy()
            yield pid, label, w, win


def spectral_features(x: np.ndarray, fs: float) -> dict:
    """
    Resume el contenido de frecuencias de una señal con 4 números (vía FFT).

    Idea sencilla: la FFT descompone la señal en ondas de distintas
    frecuencias y dice cuánta "potencia" (energía) tiene cada una.
      - Una señal periódica concentra su energía en pocas frecuencias.
      - Una señal ruidosa la reparte en muchas.

    Features que devuelve:
      peak_freq           → frecuencia (Hz) con más energía (el "ritmo" dominante)
      peak_power          → la energía de ese pico
      spectral_energy     →  la energía total (sin contar la media)
      spectral_entropy    → qué tan repartida está la energía
                            (baja = periódica/ordenada, alta = ruidosa)
    """
    x = np.asarray(x, dtype=float)
    # Se resta la media para que la FFT mida solo las FLUCTUACIONES,
    # no el nivel promedio (ese ya lo capturan los features de tiempo).
    centered = x - x.mean()

    # rfft = FFT para señales reales; devuelve solo frecuencias positivas.
    fft = np.fft.rfft(centered)
    # Frecuencias correspondientes: de 0 Hz hasta FS/2 = 5 Hz (límite de Nyquist).
    freqs = np.fft.rfftfreq(len(centered), d=1.0/fs)
    power = np.abs(fft) ** 2   # potencia = magnitud al cuadrado

    # Se excluye la componente de 0 Hz (DC), que es básicamente la media.
    freqs_nd = freqs[1:]
    power_nd = power[1:]

    total_power = power_nd.sum()
    # Si la señal es prácticamente constante, no hay nada que analizar.
    if total_power <= 1e-15:
        return {
            "peak_freq": 0.0,
            "peak_power": 0.0,
            "spectral_energy": 0.0,
            "spectral_entropy": 0.0,
        }

    peak_idx = np.argmax(power_nd)   # posición de la frecuencia más fuerte
    # Se normaliza la potencia para que sume 1 (como una distribución de probabilidad)
    # y se calcula la entropía de Shannon en bits.
    p = power_nd / total_power
    entropy = -(p * np.log2(p + 1e-15)).sum()   # +1e-15 evita log(0)

    return {
        "peak_freq": float(freqs_nd[peak_idx]),
        "peak_power": float(power_nd[peak_idx]),
        "spectral_energy": float(total_power),
        "spectral_entropy": float(entropy),
    }


def extract_features(raw: pd.DataFrame):
    """Convierte la serie de tiempo cruda en una tabla de features: una fila por ventana.

    Devuelve tres tablas:
      time_df   → 2 features × 4 métricas = 8 columnas   (dominio del tiempo)
      freq_df   → 4 features × 4 métricas = 16 columnas  (dominio de la frecuencia)
      hybrid_df → las 24 columnas juntas
    Todas incluyen además las columnas de identificación pid, class, window, start_s.
    """
    time_rows = []
    freq_rows = []

    for pid, label, w, win in iter_windows(raw):
        # Datos que identifican a la ventana (no son features, no entran al PCA).
        meta = {
            "pid": pid,
            "class": label,
            "window": w,
            "start_s": float(win["time_s"].iloc[0]),
        }

        trow = dict(meta)   # fila para la tabla de tiempo
        frow = dict(meta)   # fila para la tabla de frecuencia

        for metric in METRICS:
            x = win[metric].to_numpy()

            # Features de tiempo: nivel promedio y cuánto varía la señal.
            trow[f"{metric}_mean"] = float(np.mean(x))
            trow[f"{metric}_std"] = float(np.std(x))

            # Features de frecuencia: se agregan con el nombre de la métrica
            # como prefijo, p. ej. "CPI_peak_freq", "L2_MPKI_spectral_entropy".
            sf = spectral_features(x, FS)
            for name, value in sf.items():
                frow[f"{metric}_{name}"] = value

        time_rows.append(trow)
        freq_rows.append(frow)

    time_df = pd.DataFrame(time_rows)
    freq_df = pd.DataFrame(freq_rows)

    # La tabla híbrida se arma uniendo ambas por las columnas de identificación.
    key = ["pid", "class", "window", "start_s"]
    hybrid_df = time_df.merge(freq_df, on=key, how="inner")
    return time_df, freq_df, hybrid_df


# ---------------------------------------------------------------------
# Funciones de gráficas
# ---------------------------------------------------------------------
def savefig(name):
    """Guarda la figura actual en OUT con buena resolución y la cierra (libera memoria)."""
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_raw_examples(raw):
    """Gráficas 01 y 02: la serie de tiempo completa de CPI y de LLC_MPKI,
    tomando la PRIMERA corrida de cada clase como ejemplo representativo.
    Las clases malignas se dibujan con línea más gruesa."""
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
    """Gráfica 03: muestra UNA sola ventana de 20 s de cada clase maligna y su media,
    para ilustrar qué es exactamente lo que se resume en cada fila de features.
    Genera un archivo por clase: 03_single_window_mean_<clase>.png"""
    for label in MALICIOUS:
        pid = first_pid(raw, label)
        if pid is None:
            continue
        g = raw[raw["pid"] == pid].sort_values("time_s").reset_index(drop=True)
        win = g.iloc[:WINDOW_N]   # primeras WINDOW_N muestras = primera ventana

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(win["time_s"], win["CPI"], marker="o", markersize=2, color=COLORS[label])
        # Línea punteada horizontal = el promedio de la ventana (feature "CPI_mean").
        ax.axhline(win["CPI"].mean(), linestyle="--",
                   label=f"mean={win['CPI'].mean():.3f}")
        ax.set_title(f"One {WINDOW_S}-second CPI window — {label} (run {pid})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("CPI")
        ax.legend()
        ax.grid(alpha=0.25)
        savefig(f"03_single_window_mean_{label}.png")


def plot_same_mean_different_structure():
    """Gráficas 04 y 05 (datos INVENTADOS, solo para explicar la idea):

    Dos señales con casi la misma media y desviación estándar:
      a → una onda senoidal limpia de 0.4 Hz (periódica)
      b → ruido aleatorio
    En el dominio del tiempo (media/std) se ven iguales, pero la FFT
    las distingue claramente: la periódica tiene un pico en 0.4 Hz y el
    ruido tiene la energía repartida. Por eso agregamos features de frecuencia.
    """
    n = WINDOW_N
    t = np.arange(n) / FS
    a = 1.0 + 0.18*np.sin(2*np.pi*0.4*t)       # senoidal de 0.4 Hz
    b = 1.0 + RNG.normal(0, np.std(a), n)      # ruido con la misma std que 'a'

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, a, label=f"Periodic: mean={a.mean():.2f}, std={a.std():.2f}")
    ax.plot(t, b, label=f"Noisy: mean={b.mean():.2f}, std={b.std():.2f}", alpha=0.8)
    ax.set_title("Similar time-domain statistics, different temporal structure")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Illustrative CPI")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("04_similar_stats_different_structure.png")

    # Espectro de potencia de ambas señales.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for x, label in [(a, "Periodic"), (b, "Noisy")]:
        centered = x - x.mean()
        freq = np.fft.rfftfreq(n, 1/FS)
        power = np.abs(np.fft.rfft(centered)) ** 2
        ax.plot(freq[1:], power[1:], label=label)   # [1:] → sin la componente de 0 Hz
    ax.set_title("FFT exposes the difference")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("05_fft_periodic_vs_noise.png")


def plot_fft_examples(raw):
    """Gráfica 06: espectro de potencia del CPI en la primera ventana de cada clase.
    El eje Y es logarítmico para poder ver a la vez picos grandes y pequeños."""
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
    """Proyecta todas las ventanas a 2 dimensiones con PCA y las dibuja.

    ¿Qué es PCA? Busca las "direcciones" en las que los datos varían más.
    PC1 es la dirección con más variación, PC2 la segunda. Así podemos
    ver en un plano 2D datos que originalmente tienen 8, 16 o 24 columnas.
    Si las clases forman grupos separados en el plano, los features sí las distinguen.
    """
    # Las columnas de identificación no son features: se excluyen.
    meta_cols = {"pid", "class", "window", "start_s"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].to_numpy()
    # Estandarizar: cada feature queda con media 0 y desviación 1.
    # Sin esto, un feature con números grandes dominaría el PCA.
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xz)   # coordenadas de cada ventana en (PC1, PC2)

    # Se guardan las coordenadas en CSV para análisis posterior.
    pca_df = df[["pid", "class", "window", "start_s"]].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]
    pca_df.to_csv(OUT / filename.replace(".png", ".csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASSES:
        m = pca_df["class"] == label
        if not m.any():
            continue
        # Los ransomware se dibujan resaltados (X grande con borde negro).
        is_rw = label in MALICIOUS
        ax.scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"],
                   label=label,
                   color=COLORS[label],
                   marker="X" if is_rw else "o",
                   s=120 if is_rw else 35,
                   # borde blanco para que la X negra se distinga; negro para la roja
                   edgecolors=("white" if COLORS[label] == "#000000" else "black") if is_rw else "none",
                   alpha=0.9 if is_rw else 0.6,
                   zorder=3 if is_rw else 2)   # malignos encima de los benignos

    # Porcentaje de la variación total que explica cada componente.
    evr = pca.explained_variance_ratio_ * 100
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({evr[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1f}% variance)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.25)
    savefig(filename)

    return pca, pca_df


def plot_explained_variance(df, title, filename):
    """Gráfica de varianza acumulada: ¿cuántos componentes del PCA necesito
    para conservar, por ejemplo, el 90% de la información?
    La línea punteada marca el 90%; donde la curva la cruza es el número
    de componentes que usará luego pca_kmeans_eval."""
    meta_cols = {"pid", "class", "window", "start_s"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    Xz = StandardScaler().fit_transform(df[feature_cols])
    pca = PCA().fit(Xz)   # sin n_components → calcula todos los componentes
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100   # suma acumulada en %

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(cumulative)+1), cumulative, marker="o")
    ax.axhline(90, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.25)
    savefig(filename)


def map_clusters_to_classes(y_true, clusters):
    """K-means pone números arbitrarios a sus grupos (cluster 0, 1, 2...),
    que no tienen por qué coincidir con los números de las clases.

    Esta función encuentra el mejor emparejamiento cluster → clase usando
    el algoritmo húngaro: elige la asignación uno-a-uno que maximiza el
    número de ventanas bien clasificadas. Así podemos calcular precision,
    recall y F1 como si fuera un clasificador.
    """
    cm = confusion_matrix(y_true, clusters)
    # Se usa -cm porque linear_sum_assignment MINIMIZA el costo
    # y nosotros queremos MAXIMIZAR los aciertos.
    rows, cols = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(rows, cols)}   # cluster → clase
    y_pred = np.array([mapping[c] for c in clusters])
    return y_pred, mapping


def md_table(header, rows) -> str:
    """Arma una tabla en Markdown a partir de un encabezado y una lista de filas.
    (Se hace a mano para no depender del paquete 'tabulate'.)"""
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def pca_kmeans_eval(df, name, prefix, md):
    """Pipeline completo de agrupamiento y evaluación para un conjunto de features.

    Pasos:
      1. Estandarizar features.
      2. PCA conservando el 90% de la varianza.
      3. Método del codo (k = 1..20) para ver cuántos grupos "naturales" hay.
      4. K-means con k = número de clases, repetido con 10 semillas,
         y métricas promedio ± desviación (multiclase y maligno vs. benigno).
      5. Matriz de confusión para una corrida de referencia (semilla 42).
      6. ROC AUC: ¿qué tan bien separa la distancia a los centroides de los
         clusters malignos a las ventanas de ransomware de las benignas?

    Todo lo que se imprime también se agrega a la lista `md` (secciones en
    Markdown) que main() escribe en stats.md. Devuelve un diccionario con el
    resumen de esta familia de features para la tabla comparativa final.
    """
    meta_cols = {"pid", "class", "window", "start_s"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    Xz = StandardScaler().fit_transform(df[feature_cols].to_numpy())

    # --- Paso 2: PCA ---
    # n_components=0.90 → PCA elige automáticamente cuántos componentes
    # se necesitan para explicar al menos el 90% de la varianza.
    pca = PCA(n_components=0.90, svd_solver="full")
    Z = pca.fit_transform(Xz)
    var_pct = pca.explained_variance_ratio_.sum() * 100
    print(f"\n=== {name} ===")
    print(name, "→", pca.n_components_, "componentes,", f"{var_pct:.1f}% varianza")
    md.append(f"## Features: {name}\n")
    md.append(f"- Features de entrada: {len(feature_cols)}")
    md.append(f"- PCA: {pca.n_components_} componentes ({var_pct:.1f}% de la varianza)\n")
    # Guardar las coordenadas de cada ventana en el espacio PCA.
    pca_df = df[["pid", "class", "window", "start_s"]].copy()
    for i in range(Z.shape[1]):
        pca_df[f"PC{i+1}"] = Z[:, i]
    pca_df.to_csv(OUT / f"{prefix}_pca_scores.csv", index=False)

    # --- Paso 3: método del codo ---
    # Inercia = suma de distancias² de cada punto a su centroide.
    # Siempre baja al aumentar k; el "codo" (donde deja de bajar mucho)
    # sugiere un número razonable de grupos. La línea punteada marca
    # el número real de clases como referencia.
    ks = range(1, 21)
    inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z).inertia_ for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, inertias, marker="o")
    ax.axvline(len(CLASSES), linestyle="--", linewidth=1)
    ax.set_title(f"Elbow method — {name}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Distortion (inertia)")
    ax.grid(alpha=0.25)
    savefig(f"{prefix}_elbow.png")

    # Convertir los nombres de clase a números (0, 1, 2, ...).
    # Solo se usan las clases que realmente tienen datos.
    classes = [c for c in CLASSES if c in set(df["class"])]
    class_to_num = {c: i for i, c in enumerate(classes)}
    y_true = df["class"].map(class_to_num).to_numpy()

    # Clases malignas presentes en los datos y su versión numérica.
    mal_present = [c for c in MALICIOUS if c in class_to_num]
    mal_nums = [class_to_num[c] for c in mal_present]
    # Etiqueta binaria verdadera: 1 = ventana de ransomware (cualquiera), 0 = benigna.
    y_true_bin = np.isin(y_true, mal_nums).astype(int)

    # --- Paso 4: K-means con 10 semillas distintas ---
    # K-means depende del punto de inicio, así que se repite con varias
    # semillas para reportar promedio y desviación (más honesto que una sola corrida).
    #   precision_macro → de lo que se asignó a cada clase, cuánto era correcto (promedio entre clases)
    #   recall_macro    → de cada clase real, cuánto se recuperó (promedio entre clases)
    #   f1_macro        → balance entre precision y recall
    #   ARI             → coincidencia entre clusters y clases (1 = perfecto, ~0 = azar);
    #                     no necesita el emparejamiento húngaro
    #   silhouette      → qué tan compactos y separados son los clusters (-1 a 1);
    #                     no usa las etiquetas reales
    #   mal_precision / mal_recall / mal_f1 → detección binaria maligno vs. benigno:
    #                     una ventana se considera "detectada" si cae en un cluster
    #                     emparejado con CUALQUIER clase maligna. Así, confundir
    #                     cryptsky con ransomware NO cuenta como error de detección.
    # Nota: K-means nunca ve las etiquetas; solo se usan después para evaluar.
    rows = []
    for seed in range(10):
        km = KMeans(n_clusters=len(classes), n_init=10, random_state=seed).fit(Z)
        y_pred, mapping = map_clusters_to_classes(y_true, km.labels_)
        y_pred_bin = np.isin(y_pred, mal_nums).astype(int)
        rows.append({
            "seed": seed,
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "ARI": adjusted_rand_score(y_true, km.labels_),
            # Con ventanas cortas (1–2 s) puede haber muchísimas ventanas y el
            # silhouette es O(n²); arriba de 5000 se estima con una muestra aleatoria.
            "silhouette": silhouette_score(Z, km.labels_,
                                           sample_size=min(len(Z), 5000),
                                           random_state=seed),
            "mal_precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
            "mal_recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
            "mal_f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / f"{prefix}_kmeans_metrics.csv", index=False)
    # Imprime una tabla con el promedio y la desviación estándar de cada métrica.
    summary = metrics.drop(columns="seed").agg(["mean", "std"]).T
    print(summary.round(3))
    md.append("### K-means (10 semillas)\n")
    md.append(md_table(["Métrica", "Media", "Desv. est."],
                       [[m, r["mean"], r["std"]] for m, r in summary.iterrows()]))
    md.append("")
    result = {"features": name, "n_pc": int(pca.n_components_)}
    for m in ["f1_macro", "ARI", "silhouette", "mal_precision", "mal_recall", "mal_f1"]:
        result[m] = f"{summary.loc[m, 'mean']:.3f} ± {summary.loc[m, 'std']:.3f}"

    # --- Paso 5: matriz de confusión (corrida de referencia, semilla 42) ---
    # Filas = clase real, columnas = clase predicha (cluster ya emparejado).
    # Lo ideal es que todo caiga en la diagonal.
    km_ref = KMeans(n_clusters=len(classes), n_init=10, random_state=42).fit(Z)
    y_pred_ref, mapping_ref = map_clusters_to_classes(y_true, km_ref.labels_)
    cm = confusion_matrix(y_true, y_pred_ref)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")   # celdas más oscuras = más ventanas
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    # Nombres de las clases malignas en rojo para ubicarlas rápido.
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        if tick.get_text() in MALICIOUS:
            tick.set_color("red")
            tick.set_fontweight("bold")

    # Escribe el número dentro de cada celda (blanco si el fondo es oscuro).
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted class (K-means, mapped)")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix — {name}")
    savefig(f"{prefix}_confusionmatrix.png")

    # --- Paso 6: ROC AUC de los ransomware contra las cargas benignas ---
    # Idea: si una ventana está cerca del centroide de algún cluster maligno,
    # es "sospechosa". Medimos qué tan bien esa cercanía separa
    # ransomware (1) de benigno (0).
    #   AUC = 1.0 → separación perfecta
    #   AUC = 0.5 → igual que adivinar al azar
    if not mal_present:
        print("[aviso] no hay clases malignas en los datos; se omite el ROC AUC")
        md.append("_No hay clases malignas en los datos; se omite el ROC AUC._\n")
        return result

    # Cluster que quedó emparejado con cada clase (clase numérica → cluster).
    class_to_cluster = {cls: c for c, cls in mapping_ref.items()}

    # Distancia euclidiana de cada ventana al centroide de cada cluster maligno.
    dists = {}
    for cname in mal_present:
        center = km_ref.cluster_centers_[class_to_cluster[class_to_num[cname]]]
        dists[cname] = np.linalg.norm(Z - center, axis=1)

    benign = y_true_bin == 0
    auc_rows = []

    # (a) AUC por cada ransomware, contra SOLO las ventanas benignas
    #     (se excluye al otro ransomware para que no cuente como "negativo").
    for cname in mal_present:
        m = benign | (y_true == class_to_num[cname])
        y_bin = (y_true[m] == class_to_num[cname]).astype(int)
        # El score debe ser MAYOR para lo más sospechoso → distancia con signo negativo.
        auc = roc_auc_score(y_bin, -dists[cname][m])
        print(f"ROC AUC {cname} vs benignos: {auc:.3f}")
        auc_rows.append([f"{cname} vs benignos", auc])
        result[f"AUC {cname}"] = f"{auc:.3f}"

    # (b) AUC combinado: cualquier ransomware vs. benignos.
    #     Score = distancia al centroide maligno MÁS CERCANO.
    min_dist = np.min(np.vstack(list(dists.values())), axis=0)
    auc_all = roc_auc_score(y_true_bin, -min_dist)
    label_all = f"malignos ({' + '.join(mal_present)}) vs benignos"
    print(f"ROC AUC {label_all}: {auc_all:.3f}")
    auc_rows.append([label_all, auc_all])
    result["AUC malignos"] = f"{auc_all:.3f}"

    md.append("### ROC AUC (distancia al centroide maligno, semilla 42)\n")
    md.append(md_table(["Comparación", "AUC"], auc_rows))
    md.append("")
    return result


# ---------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------
def run_window(raw: pd.DataFrame, window_s: int):
    """Corre el pipeline completo para UN tamaño de ventana y guarda todo en su carpeta.
    Devuelve la lista de resúmenes (time, frequency, hybrid) para la tabla global."""
    set_window(window_s)
    print(f"\n{'#' * 60}\n# Ventana de {WINDOW_S} s ({WINDOW_N} muestras) → {OUT}\n{'#' * 60}")

    # 1) Tabla unificada de telemetría (la misma en todas las carpetas).
    raw.to_csv(OUT / "pcm_telemetry.csv", index=False)

    # 2) Cortar en ventanas y calcular features (tiempo, frecuencia, híbrido).
    print("Extracting window features...")
    time_df, freq_df, hybrid_df = extract_features(raw)
    time_df.to_csv(OUT / "features_time_domain.csv", index=False)
    freq_df.to_csv(OUT / "features_frequency_domain.csv", index=False)
    hybrid_df.to_csv(OUT / "features_hybrid.csv", index=False)

    # 3) Gráficas explicativas de los datos crudos y de la FFT.
    print("Generating explanatory charts...")
    plot_raw_examples(raw)
    plot_window_example(raw)
    plot_same_mean_different_structure()
    plot_fft_examples(raw)

    # 4) ¿Cuántos componentes de PCA se necesitan? (híbrido y frecuencia)
    plot_explained_variance(
        hybrid_df,
        "PCA cumulative explained variance — hybrid features",
        "07_hybrid_pca_explained_variance.png",
    )

    plot_explained_variance(
        freq_df,
        "PCA cumulative explained variance — frequency features",
        "07_frequency_pca_explained_variance.png",
    )

    # 5) Vista 2D de las ventanas usando solo features de tiempo.
    print("Generating time-domain PCA plot...")
    pca_plot(
        time_df,
        "PCA — time-domain features (mean + standard deviation)",
        "08_pca_time_domain.png",
    )

    # 6) Agrupamiento y evaluación para cada tipo de features,
    #    para comparar cuál separa mejor las clases.
    print("PCA (90%) + K-means...")

    # Encabezado de stats.md: contexto de la corrida.
    counts = time_df["class"].value_counts()
    md = [
        "# Estadísticas — PCM, PCA + K-means\n",
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
        pca_kmeans_eval(time_df, "time", "08_time", md),
        pca_kmeans_eval(freq_df, "frequency", "09_freq", md),
        pca_kmeans_eval(hybrid_df, "hybrid", "10_hybrid", md),
    ]

    # Tabla comparativa final: una fila por familia de features (media ± desv.).
    cols = list(results[0].keys())
    md.append("## Comparación entre familias de features\n")
    md.append(md_table(cols, [[r.get(c, "") for c in cols] for r in results]))
    md.append("")

    (OUT / "stats.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Estadísticas guardadas en {OUT / 'stats.md'}")
    return results


def main():
    BASE_OUT.mkdir(exist_ok=True)

    # Cargar los CSV una sola vez. Se exige que cada corrida alcance al menos
    # para una ventana del tamaño MÁS GRANDE, para usar las mismas corridas siempre.
    print("Loading PCM telemetry...")
    raw = load_pcm_dataset(min_samples=int(FS * max(WINDOW_SIZES_S)))

    all_results = []
    for window_s in WINDOW_SIZES_S:
        for r in run_window(raw, window_s):
            all_results.append({"window_s": window_s, **r})

    # Resumen global: una fila por (ventana, familia de features), para ver
    # de un vistazo cómo cambia el desempeño con el tamaño de ventana.
    cols = ["window_s"] + [c for c in all_results[0] if c != "window_s"]
    md = [
        "# Comparación entre tamaños de ventana\n",
        f"- Generado: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Ventanas evaluadas: {', '.join(f'{w} s' for w in WINDOW_SIZES_S)}",
        "- El detalle de cada ventana está en `window_XXs/stats.md`.\n",
        md_table(cols, [[r.get(c, "") for c in cols] for r in all_results]),
        "",
    ]
    (BASE_OUT / "stats_all_windows.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nDone. Outputs written to: {BASE_OUT.resolve()}")


# Esto hace que main() solo corra cuando ejecutas el archivo directamente
# (y no cuando lo importas desde otro script).
if __name__ == "__main__":
    main()
