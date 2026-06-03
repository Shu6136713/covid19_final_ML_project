"""Generate static figures for README preview (no model training)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

PALETTE = {"survived": "#3498db", "died": "#e74c3c"}
MODEL_COLORS = ["#2ecc71", "#9b59b6", "#f39c12"]
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "white"})


def load_cleaned_df():
    df = pd.read_csv(ROOT / "data" / "Covid_Data.csv")
    df.rename(
        columns={
            "SEX": "GENDER",
            "MEDICAL_UNIT": "MEDICAL_TYPE",
            "PATIENT_TYPE": "TREATMENT_TYPE",
            "CLASIFFICATION_FINAL": "COVID_RES",
        },
        inplace=True,
    )
    df["DEATH"] = (df["DATE_DIED"] != "9999-99-99").astype(int)
    df.drop(columns="DATE_DIED", inplace=True)
    bool_cols = [
        "INTUBED", "PNEUMONIA", "GENDER", "PREGNANT", "DIABETES", "COPD", "ASTHMA",
        "INMSUPR", "HIPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR", "OBESITY",
        "RENAL_CHRONIC", "TOBACCO", "ICU",
    ]
    for col in bool_cols:
        df[col] = df[col].map({1: 1, 2: 0})
    df["COVID_RES"] = df["COVID_RES"].apply(lambda x: 0 if x >= 4 else x)
    missing_pct = df.isnull().mean()
    df.drop(columns=missing_pct[missing_pct > 0.40].index.tolist(), inplace=True)
    df.dropna(inplace=True)
    return df


def fig_eda(df):
    y = df["DEATH"]
    labels = ["Survived", "Died"]
    colors = [PALETTE["survived"], PALETTE["died"]]
    counts = y.value_counts().sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("Target Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Patients")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 15000, f"{v:,}\n({v/len(y):.1%})", ha="center", fontsize=9)

    age_death = df.groupby(pd.cut(df["AGE"], bins=range(0, 105, 10), right=False))["DEATH"].mean()
    age_death.plot(kind="bar", ax=axes[1], color=PALETTE["died"], edgecolor="white")
    axes[1].set_title("Death Rate by Age Group", fontweight="bold")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Death rate")
    axes[1].tick_params(axis="x", rotation=45)

    num_cols = [c for c in df.columns if c != "DEATH" and df[c].dtype in ("int64", "float64")]
    corr = df[num_cols + ["DEATH"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.6,
        square=True, linewidths=0.5, ax=axes[2], cbar_kws={"shrink": 0.8},
    )
    axes[2].set_title("Feature Correlations", fontweight="bold")
    plt.suptitle("COVID-19 Dataset Overview", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    out = FIG_DIR / "eda_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def fig_model_comparison():
    results = pd.DataFrame(
        {
            "Accuracy": [0.8931, 0.9034, 0.9319],
            "Precision": [0.3988, 0.4140, 0.5404],
            "Recall": [0.9207, 0.7846, 0.4418],
            "F1-Score": [0.5566, 0.5420, 0.4861],
            "ROC-AUC": [0.9531, 0.9345, 0.8974],
        },
        index=["Logistic Regression", "Random Forest", "KNN"],
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    results.T.plot(kind="bar", ax=ax, rot=0, color=MODEL_COLORS, edgecolor="white", linewidth=1.2)
    ax.set_title("Model Comparison — Validation Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    out = FIG_DIR / "model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def fig_results_summary():
    test_metrics = {
        "Accuracy": 0.9011, "Precision": 0.4168, "Recall": 0.8952,
        "F1-Score": 0.5688, "ROC-AUC": 0.9539,
    }
    results = pd.DataFrame(
        {"Recall": [0.9207, 0.7846, 0.4418], "F1-Score": [0.5566, 0.5420, 0.4861], "ROC-AUC": [0.9531, 0.9345, 0.8974]},
        index=["Logistic Regression", "Random Forest", "KNN"],
    )
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    plot_df = results
    x = np.arange(len(plot_df))
    w = 0.25
    for i, col in enumerate(plot_df.columns):
        ax0.bar(x + (i - 1) * w, plot_df[col], width=w, label=col, color=MODEL_COLORS[i], edgecolor="white")
    ax0.set_xticks(x)
    ax0.set_xticklabels(plot_df.index, rotation=15, ha="right")
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("Score")
    ax0.set_title("Validation — Key Metrics", fontweight="bold")
    ax0.legend(loc="lower right", fontsize=9)

    ax1 = fig.add_subplot(gs[0, 1])
    names = list(test_metrics.keys())
    vals = list(test_metrics.values())
    bars = ax1.barh(names, vals, color=MODEL_COLORS[0], edgecolor="white")
    ax1.set_xlim(0, 1.05)
    ax1.set_title("Test Set — Logistic Regression\n(threshold=0.6)", fontweight="bold")
    for b, v in zip(bars, vals):
        ax1.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ax2.text(
        0.5, 0.5,
        "Selected model\nLogistic Regression\n\nValidation ROC-AUC\n0.9531\n\nTest ROC-AUC\n0.9539\n\nTest Recall\n0.8952",
        ha="center", va="center", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", edgecolor="#bdc3c7"),
    )
    ax2.set_title("Final Selection", fontweight="bold")
    plt.suptitle("COVID-19 Mortality Prediction — Results Summary", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "results_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    df = load_cleaned_df()
    fig_eda(df)
    fig_model_comparison()
    fig_results_summary()
