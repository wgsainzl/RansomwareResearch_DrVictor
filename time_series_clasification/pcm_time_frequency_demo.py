#!/usr/bin/env python3
"""
Educational demo:
Per-PID microarchitectural time-series characterization in
time domain, frequency domain, and PCA.

IMPORTANT:
The generated classes are synthetic. They illustrate the methodology
and are NOT measured ransomware signatures.

Dependencies:
    pip install numpy pandas matplotlib scikit-learn

Run:
    python pcm_time_frequency_demo.py

Outputs are written to ./pcm_demo_output/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             adjusted_rand_score, silhouette_score, roc_auc_score)
from scipy.optimize import linear_sum_assignment


RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_DIR = Path.home() / "Desktop" / "Resultados_ALL"

FS = 10.0                 # samples/second
#DURATION_S = 60           # seconds per synthetic PID
WINDOW_S = 20              # seconds/window
WINDOW_N = int(FS * WINDOW_S)
#N_PIDS_PER_CLASS = 5

METRICS = ["CPI", "IPC", "LLC_MPKI", "L2_MPKI"]
CLASSES = ["7zip", "ffmpeg", "gpg", "openssl", "readwrite", "rsync", "stress", "sysbench", "mlc", "cryptsky"]


OUT = Path("pcm_demo_output")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# PCM telemetry loader
# ---------------------------------------------------------------------
def load_pcm_csv(path: Path, label: str, run_id: int) -> pd.DataFrame:
    """Lee un CSV de pcm (2 filas de encabezado) y devuelve el formato largo."""
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

    # Rellenar huecos (NaN/inf) interpolando para no romper el espaciado temporal
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.interpolate(limit_direction="both")

    out.insert(0, "time_s", np.arange(len(out)) / FS)
    out.insert(1, "pid", run_id)          # "pid" = id de corrida
    out.insert(2, "class", label)
    out["run_file"] = path.name
    return out


def load_pcm_dataset() -> pd.DataFrame:
    frames = []
    run_id = 1000
    for label in CLASSES:
        files = sorted((DATA_DIR / label).glob("*pcm*.csv"))
        if not files:
            print(f"[aviso] sin CSV para '{label}' en {DATA_DIR / label}")
            continue
        for f in files:
            d = load_pcm_csv(f, label, run_id)
            if len(d) < WINDOW_N:
                print(f"[aviso] {f.name}: {len(d)} muestras (< {WINDOW_N}), se omite")
                continue
            frames.append(d)
            run_id += 1
    return pd.concat(frames, ignore_index=True)
# ---------------------------------------------------------------------
# Windowing and feature extraction
# ---------------------------------------------------------------------
def iter_windows(df: pd.DataFrame):
    for (pid, label), g in df.groupby(["pid", "class"], sort=False):
        g = g.sort_values("time_s").reset_index(drop=True)
        n_windows = len(g) // WINDOW_N

        for w in range(n_windows):
            start = w * WINDOW_N
            stop = start + WINDOW_N
            win = g.iloc[start:stop].copy()
            yield pid, label, w, win


def spectral_features(x: np.ndarray, fs: float) -> dict:
    """
    Compute compact FFT-based features after mean removal.
    Uses a one-sided real FFT.
    """
    x = np.asarray(x, dtype=float)
    centered = x - x.mean()

    fft = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0/fs)
    power = np.abs(fft) ** 2

    # Exclude DC (0 Hz).
    freqs_nd = freqs[1:]
    power_nd = power[1:]

    total_power = power_nd.sum()
    if total_power <= 1e-15:
        return {
            "peak_freq": 0.0,
            "log_peak_power": 0.0,
            "log_spectral_energy": 0.0,
            "spectral_entropy": 0.0,
        }

    peak_idx = np.argmax(power_nd)
    p = power_nd / total_power
    entropy = -(p * np.log2(p + 1e-15)).sum()

    return {
        "peak_freq": float(freqs_nd[peak_idx]),
        "log_peak_power": float(np.log10(power_nd[peak_idx] + 1)),
        "log_spectral_energy": float(np.log10(total_power + 1)),
        "spectral_entropy": float(entropy),
    }


def extract_features(raw: pd.DataFrame):
    time_rows = []
    freq_rows = []

    for pid, label, w, win in iter_windows(raw):
        meta = {
            "pid": pid,
            "class": label,
            "window": w,
            "start_s": float(win["time_s"].iloc[0]),
        }

        trow = dict(meta)
        frow = dict(meta)

        for metric in METRICS:
            x = win[metric].to_numpy()

            # Time-domain features.
            trow[f"{metric}_log_mean"] = float(np.log10(np.mean(x) + 1))
            trow[f"{metric}_log_std"]  = float(np.log10(np.std(x) + 1))

            # Frequency-domain features.
            sf = spectral_features(x, FS)
            for name, value in sf.items():
                frow[f"{metric}_{name}"] = value

        time_rows.append(trow)
        freq_rows.append(frow)

    time_df = pd.DataFrame(time_rows)
    freq_df = pd.DataFrame(freq_rows)

    key = ["pid", "class", "window", "start_s"]
    hybrid_df = time_df.merge(freq_df, on=key, how="inner")
    return time_df, freq_df, hybrid_df


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()


def plot_raw_examples(raw):
    # One representative PID per class.
    fig, ax = plt.subplots(figsize=(11, 5))
    for label in CLASSES:
        pid = raw.loc[raw["class"] == label, "pid"].iloc[0]
        g = raw[raw["pid"] == pid]
        ax.plot(g["time_s"], g["CPI"], label=f"{label} (run {pid})", linewidth=1.2)
    ax.set_title("PCM CPI time series (System)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CPI")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("01_raw_cpi_timeseries.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    for label in CLASSES:
        pid = raw.loc[raw["class"] == label, "pid"].iloc[0]
        g = raw[raw["pid"] == pid]
        ax.plot(g["time_s"], g["LLC_MPKI"], label=f"{label} (run {pid})", linewidth=1.2)
    ax.set_title("PCM LLC MPKI time series (System)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LLC MPKI")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("02_raw_llc_mpki_timeseries.png")


def plot_window_example(raw):
    label = "cryptsky"
    pid = raw.loc[raw["class"] == label, "pid"].iloc[0]
    g = raw[raw["pid"] == pid].sort_values("time_s").reset_index(drop=True)
    win = g.iloc[:WINDOW_N]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(win["time_s"], win["CPI"], marker="o", markersize=2)
    ax.axhline(win["CPI"].mean(), linestyle="--",
               label=f"mean={win['CPI'].mean():.3f}")
    ax.set_title(f"One {WINDOW_S}-second CPI window — {label} (run {pid})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CPI")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("03_single_window_mean.png")


def plot_same_mean_different_structure():
    # Similar mean/std, different organization in time.
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
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in CLASSES:
        pid = raw.loc[raw["class"] == label, "pid"].iloc[0]
        g = raw[raw["pid"] == pid].sort_values("time_s").reset_index(drop=True)
        x = g.iloc[:WINDOW_N]["CPI"].to_numpy()
        centered = x - x.mean()
        freq = np.fft.rfftfreq(len(x), 1/FS)
        power = np.abs(np.fft.rfft(centered)) ** 2
        ax.plot(freq[1:], power[1:], label=label)
    ax.set_title(f"CPI power spectra for one {WINDOW_S}-second window")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig("06_cpi_power_spectra.png")


def pca_plot(df, title, filename):
    meta_cols = {"pid", "class", "window", "start_s"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].to_numpy()
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xz)

    pca_df = df[["pid", "class", "window", "start_s"]].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]
    pca_df.to_csv(OUT / filename.replace(".png", ".csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASSES:
        m = pca_df["class"] == label
        if not m.any():
            continue
        is_rw = (label == "cryptsky")
        ax.scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"],
                   label=label,
                   marker="X" if is_rw else "o",
                   s=120 if is_rw else 35,
                   edgecolors="black" if is_rw else "none",
                   alpha=0.9 if is_rw else 0.6)

    evr = pca.explained_variance_ratio_ * 100
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({evr[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1f}% variance)")
    ax.legend()
    ax.grid(alpha=0.25)
    savefig(filename)

    return pca, pca_df


def plot_explained_variance(df, title, filename):
    meta_cols = {"pid", "class", "window", "start_s"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    Xz = StandardScaler().fit_transform(df[feature_cols])
    pca = PCA().fit(Xz)
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100

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
    cm = confusion_matrix(y_true, clusters)
    rows, cols = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(rows, cols)}
    y_pred = np.array([mapping[c] for c in clusters])
    return y_pred, mapping


def pca_kmeans_eval(df, name, prefix):
    meta_cols = {"pid", "class", "window", "start_s"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    Xz = StandardScaler().fit_transform(df[feature_cols].to_numpy())

    pca = PCA(n_components=0.90, svd_solver="full")
    Z = pca.fit_transform(Xz)
    print(name, "→", pca.n_components_, "componentes,",
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% varianza")
    pca_df = df[["pid", "class", "window", "start_s"]].copy()
    for i in range(Z.shape[1]):
        pca_df[f"PC{i+1}"] = Z[:, i]
    pca_df.to_csv(OUT / f"{prefix}_pca_scores.csv", index=False)
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

    classes = [c for c in CLASSES if c in set(df["class"])]
    class_to_num = {c: i for i, c in enumerate(classes)}
    y_true = df["class"].map(class_to_num).to_numpy()

    rows = []
    for seed in range(10):
        km = KMeans(n_clusters=len(classes), n_init=10, random_state=seed).fit(Z)
        y_pred, mapping = map_clusters_to_classes(y_true, km.labels_)
        rows.append({
            "seed": seed,
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "ARI": adjusted_rand_score(y_true, km.labels_),
            "silhouette": silhouette_score(Z, km.labels_),
        })
    metrics = pd.DataFrame(rows)
    print(metrics.drop(columns="seed").agg(["mean", "std"]).round(3))
    
    km_ref = KMeans(n_clusters=len(classes), n_init=10, random_state=42).fit(Z)
    y_pred_ref, mapping_ref = map_clusters_to_classes(y_true, km_ref.labels_)
    cm = confusion_matrix(y_true, y_pred_ref)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted class (K-means, mapped)")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix — {name}")
    savefig(f"{prefix}_confusionmatrix.png")

    # E1. ¿Qué número de cluster quedó mapeado a cryptsky?
    #     mapping_ref es {cluster: clase}; necesitas el cluster cuya clase sea la de cryptsky
    target = class_to_num["cryptsky"]
    c_rw = [c for c, cls in mapping_ref.items() if cls == target][0]
    
    # E2. Distancia de cada ventana a ese centroide
    center = km_ref.cluster_centers_[c_rw]
    dist = np.linalg.norm(Z - center, axis=1)
    
    # E3. Etiqueta binaria: 1 si la ventana es cryptsky, 0 si no
    y_bin = (y_true == target).astype(int)
    
    # E4. AUC: el score debe ser MAYOR para lo más sospechoso
    auc = roc_auc_score(y_bin, -dist)
    print(f"ROC AUC cryptsky vs resto: {auc:.3f}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading PCM telemetry...")
    raw = load_pcm_dataset()
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

    print("Generating time-domain PCA plot...")
    pca_plot(
        time_df,
        "PCA — time-domain features (log mean + log standard deviation)",
        "08_pca_time_domain.png",
    )
    
    print("PCA (90%) + K-means...")

    pca_kmeans_eval(time_df, "time", "08_time")
    
    pca_kmeans_eval(freq_df, "frequency", "09_freq")
    
    pca_kmeans_eval(hybrid_df, "hybrid", "10_hybrid")
    
    print(f"\nDone. Outputs written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()

