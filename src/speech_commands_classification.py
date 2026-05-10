#!/usr/bin/env python3
"""
speech_commands_classification.py
Assignment: reconocimiento de palabras con MFCC + SVM / MLP (+ PCA)

Uso básico:
    python speech_commands_classification.py
    python speech_commands_classification.py --root data --classes happy right follow on forward
"""

import argparse
import pathlib
import time
import warnings

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",    default="data",
                   help="Directorio donde se descarga SpeechCommands")
    p.add_argument("--classes", nargs="+",
                   default=["happy", "right", "follow", "on", "forward"],
                   help="Palabras a reconocer")
    p.add_argument("--n_mfcc",  type=int, default=40,
                   help="Número de coeficientes MFCC")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Folds para cross-validation")
    p.add_argument("--pca_dims", nargs="*", default=["0", "50", "100", "0.95"],
                   help="Dimensiones PCA. Use '0' para sin PCA, "
                        "int para n_componentes, float<1 para varianza retenida.")
    return p.parse_args()


def load_dataset(root: str, classes: list[str], target_sr: int = 16_000):
    """
    Descarga SpeechCommands (train+val+test) y devuelve waveforms con
    longitud fija (padding con ceros / recorte centrado).
    """
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    splits = []
    for subset in ("training", "validation", "testing"):
        splits.extend(
            torchaudio.datasets.SPEECHCOMMANDS(root, subset=subset, download=True)
        )

    class_set = set(classes)
    raw, labels = [], []

    print(f"  Cargando {len(splits):,} entradas totales del dataset …")
    for waveform, sr, label, *_ in splits:
        if label not in class_set:
            continue
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        raw.append(waveform)
        labels.append(label)

    max_len = max(w.shape[-1] for w in raw)
    print(f"  Muestras seleccionadas: {len(labels):,}   "
          f"Longitud máxima: {max_len} samples ({max_len/target_sr:.3f} s)")

    padded = []
    for w in raw:
        diff = max_len - w.shape[-1]
        if diff > 0:
            pad_l = diff // 2
            pad_r = diff - pad_l
            w = F.pad(w, (pad_l, pad_r))
        elif diff < 0:
            start = (-diff) // 2
            w = w[:, start : start + max_len]
        padded.append(w)

    return padded, labels, max_len


def build_features(waveforms: list, n_mfcc: int, target_sr: int = 16_000) -> np.ndarray:
    """
    Convierte cada waveform a un vector MFCC aplanado (1D).
    Todos los vectores tienen la misma longitud porque los waveforms
    fueron estandarizados previamente.

    Forma de cada MFCC: (n_mfcc, T)  →  aplanado: (n_mfcc * T,)
    """
    mfcc_tf = torchaudio.transforms.MFCC(
        sample_rate=target_sr,
        n_mfcc=n_mfcc,
        melkwargs=dict(n_fft=1024, hop_length=512, n_mels=64),
    )
    feats = []
    with torch.no_grad():
        for w in waveforms:
            m = mfcc_tf(w).squeeze(0)
            feats.append(m.flatten().numpy())

    X = np.vstack(feats)
    print(f"  Dimensión de cada vector MFCC: {X.shape[1]:,}")
    return X


def make_pipeline(estimator, pca_dim, seed: int) -> Pipeline:
    """
    Construye: StandardScaler → [PCA opcional] → Clasificador

    pca_dim:
        None / 0  → sin PCA
        int > 0   → n_components exacto
        float < 1 → varianza retenida (ej. 0.95)
    """
    steps = [("scaler", StandardScaler())]

    if pca_dim and pca_dim != 0:
        solver = "randomized" if isinstance(pca_dim, int) else "full"
        steps.append(("pca", PCA(n_components=pca_dim,
                                  svd_solver=solver,
                                  random_state=seed)))

    steps.append(("clf", estimator))
    return Pipeline(steps)


def get_models(seed: int) -> dict:
    return {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=seed),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=seed,
        ),
    }


def run_experiments(X, y, pca_dims, seed, cv_folds, class_names):
    """
    Para cada combinación (modelo × pca_dim):
        - Cross-validation en train
        - Fit final + test accuracy + tiempo
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    records   = []
    cms       = {}

    for pca_dim in pca_dims:
        pca_label = "Sin PCA" if not pca_dim else str(pca_dim)

        for model_name, estimator in get_models(seed).items():
            pipe = make_pipeline(estimator, pca_dim, seed)

            t0 = time.perf_counter()
            cv_scores = cross_val_score(pipe, X_train, y_train,
                                        cv=cv, scoring="accuracy", n_jobs=-1)
            cv_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            real_components = X.shape[1]
            if "pca" in pipe.named_steps:
                real_components = pipe.named_steps["pca"].n_components_

            t0 = time.perf_counter()
            y_pred    = pipe.predict(X_test)
            test_time = (time.perf_counter() - t0) * 1000

            acc_test = (y_pred == y_test).mean()

            records.append(dict(
                Modelo       = model_name,
                PCA          = pca_label,
                Componentes  = real_components,
                Acc_CV_mean  = round(cv_scores.mean(), 4),
                Acc_CV_std   = round(cv_scores.std(),  4),
                Acc_Test     = round(acc_test,          4),
                Train_s      = round(train_time,        2),
                Test_ms      = round(test_time,         1),
                CV_s         = round(cv_time,           2),
            ))

            if not pca_dim:
                cms[model_name] = (y_test, y_pred)

            print(f"  [{model_name:3s} | PCA={pca_label:>6s}]  "
                  f"CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}  "
                  f"Test={acc_test:.3f}  "
                  f"Train={train_time:.1f}s  Test={test_time:.1f}ms")

    df = pd.DataFrame(records)
    return df, cms, y_test, class_names


def plot_pca_variance(X, seed, out_dir: pathlib.Path):
    """
    Ajusta PCA completo y grafica varianza acumulada vs. componentes.
    Marca líneas en 90 %, 95 % y 99 %.
    """
    print("  Calculando varianza explicada por PCA …")
    pca = PCA(svd_solver="full", random_state=seed)
    scaler = StandardScaler()
    pca.fit(scaler.fit_transform(X))

    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(cum_var) + 1), cum_var, linewidth=1.5, color="#2196F3")
    for threshold, color in [(90, "#FF9800"), (95, "#F44336"), (99, "#9C27B0")]:
        idx = np.searchsorted(cum_var, threshold)
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1,
                   label=f"{threshold}% → {idx} componentes")
        ax.axvline(idx, color=color, linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlabel("Número de componentes principales")
    ax.set_ylabel("Varianza explicada acumulada (%)")
    ax.set_title("Varianza acumulada explicada por PCA")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "pca_varianza.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardada: {path}")


def plot_results(df: pd.DataFrame, out_dir: pathlib.Path):
    """
    1) Accuracy (CV y Test) por modelo y configuración de PCA
    2) Tiempos de entrenamiento por modelo y configuración de PCA
    """
    palette = {"SVM": "#2196F3", "MLP": "#F44336"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, metric, title in zip(
        axes,
        ["Acc_CV_mean", "Acc_Test"],
        ["Accuracy – Cross-Validation (media ± std)", "Accuracy – Test set"],
    ):
        for model, grp in df.groupby("Modelo"):
            x = range(len(grp))
            ax.bar(
                [xi + (0.35 if model == "MLP" else 0) for xi in x],
                grp[metric],
                width=0.35,
                label=model,
                color=palette[model],
                alpha=0.85,
            )
            if metric == "Acc_CV_mean":
                ax.errorbar(
                    [xi + (0.35 if model == "MLP" else 0) for xi in x],
                    grp[metric],
                    yerr=grp["Acc_CV_std"],
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                    linewidth=1.2,
                )

        ax.set_xticks([xi + 0.175 for xi in range(len(df["PCA"].unique()))])
        ax.set_xticklabels(df["PCA"].unique(), rotation=15)
        ax.set_xlabel("Configuración PCA")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "accuracy_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardada: {path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, grp in df.groupby("Modelo"):
        x = range(len(grp))
        ax.bar(
            [xi + (0.35 if model == "MLP" else 0) for xi in x],
            grp["Train_s"],
            width=0.35,
            label=model,
            color=palette[model],
            alpha=0.85,
        )
        for xi, val in zip(x, grp["Train_s"]):
            ax.text(xi + (0.35 if model == "MLP" else 0),
                    val + 0.3, f"{val:.1f}s",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks([xi + 0.175 for xi in range(len(df["PCA"].unique()))])
    ax.set_xticklabels(df["PCA"].unique(), rotation=15)
    ax.set_xlabel("Configuración PCA")
    ax.set_ylabel("Tiempo de entrenamiento (s)")
    ax.set_title("Tiempo de entrenamiento por modelo y configuración PCA")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "training_time_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardada: {path}")


def plot_confusion_matrices(cms: dict, class_names: list, out_dir: pathlib.Path):
    """
    Grafica la matriz de confusión para cada modelo (sin PCA).
    """
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (model_name, (y_true, y_pred)) in zip(axes, cms.items()):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Matriz de confusión – {model_name} (sin PCA)")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardada: {path}")


def main():
    args = parse_args()
    out_dir = pathlib.Path("outputs")
    out_dir.mkdir(exist_ok=True)

    def parse_dim(s: str):
        if s in ("0", "None", "none", ""):
            return None
        try:
            val = float(s)
            return int(val) if val >= 1 else val
        except ValueError:
            return None

    pca_dims = [parse_dim(d) for d in args.pca_dims]

    print("\n1) Cargando dataset …")
    waveforms, labels, max_len = load_dataset(args.root, args.classes)

    le = LabelEncoder()
    y  = le.fit_transform(labels)
    class_names = list(le.classes_)
    print(f"  Clases: {class_names}")
    print(f"  Distribución: { {c: (np.array(labels)==c).sum() for c in class_names} }")

    print("\n2) Extrayendo features MFCC …")
    X = build_features(waveforms, args.n_mfcc)

    print("\n3) Analizando varianza PCA …")
    plot_pca_variance(X, args.seed, out_dir)

    print(f"\n4) Ejecutando experimentos "
          f"(CV={args.cv_folds} folds, PCA dims={pca_dims}) …")
    df, cms, y_test, _ = run_experiments(
        X, y, pca_dims, args.seed, args.cv_folds, class_names
    )

    print("\n5) Resultados completos:")
    print(df.to_markdown(index=False))

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV guardado: {csv_path.resolve()}")

    print("\n6) Generando gráficas …")
    plot_results(df, out_dir)
    plot_confusion_matrices(cms, class_names, out_dir)

    print(f"\n✓ Todo listo. Archivos en: {out_dir.resolve()}/")


if __name__ == "__main__":
    main()