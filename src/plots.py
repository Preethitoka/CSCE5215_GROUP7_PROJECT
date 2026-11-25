"""Plot helpers shared by run_all.py, error_analysis.py and focused.py."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion(C, labels, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, C[i, j], ha="center", va="center",
                    color="white" if C[i, j] > C.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_subject(per_method, path):
    methods = list(per_method)
    subjects = sorted(next(iter(per_method.values())))
    x = np.arange(len(subjects))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(methods):
        ax.bar(x + i * w, [per_method[m][s] for s in subjects], w, label=m)
    ax.set_xticks(x + w * (len(methods) - 1) / 2, [str(s) for s in subjects])
    ax.set_xlabel("Test participant")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.set_title("Per-participant macro-F1 (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_focused(ks, means, stds, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("Number of training participants (k)")
    ax.set_ylabel("Test macro-F1 (mean ± std, 10 draws)")
    ax.set_title("SoftMax: participant diversity at 600 windows")
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
