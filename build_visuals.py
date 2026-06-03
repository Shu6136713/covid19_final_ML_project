"""
Generate all README figures + embed PNG outputs into modeling.ipynb for GitHub preview.
Uses a stratified sample for fast training (figures match notebook style; run notebook for exact numbers).
"""
import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
NB = ROOT / "modeling.ipynb"
FIG.mkdir(exist_ok=True)

PALETTE = {"survived": "#3498db", "died": "#e74c3c"}
MODEL_COLORS = ["#2ecc71", "#9b59b6", "#f39c12"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "KNN"]
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "white"})


def load_cleaned_df(max_rows=None):
    df = pd.read_csv(ROOT / "data" / "Covid_Data.csv")
    df.rename(columns={
        "SEX": "GENDER", "MEDICAL_UNIT": "MEDICAL_TYPE",
        "PATIENT_TYPE": "TREATMENT_TYPE", "CLASIFFICATION_FINAL": "COVID_RES",
    }, inplace=True)
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
    if max_rows and len(df) > max_rows:
        idx_train, _, _, _ = train_test_split(
            np.arange(len(df)),
            df["DEATH"],
            train_size=max_rows,
            stratify=df["DEATH"],
            random_state=42,
        )
        df = df.iloc[idx_train].reset_index(drop=True)
    return df


def train_all(df):
    X = df.drop(columns="DEATH")
    y = df["DEATH"]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)

    models = {}
    for name, m, Xv, scaled in [
        ("Logistic Regression", lr, X_val_s, True),
        ("Random Forest", rf, X_val, False),
        ("KNN", knn, X_val_s, True),
    ]:
        prob = m.predict_proba(Xv)[:, 1]
        pred = (prob >= 0.5).astype(int)
        models[name] = {"pred": pred, "prob": prob, "model": m}
    return y_val, models


def save_fig(fig, name):
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", path)
    return path


def fig_roc(y_val, models):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, d) in enumerate(models.items()):
        RocCurveDisplay.from_predictions(y_val, d["prob"], name=name, ax=ax)
        ax.lines[-1].set_color(MODEL_COLORS[i])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
    ax.set_title("ROC Curves — Validation Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(fig, "roc_curves.png")


def fig_confusion(y_val, models):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (name, d), c in zip(axes, models.items(), MODEL_COLORS):
        cm = confusion_matrix(y_val, d["pred"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="PuBuGn", ax=ax,
                    xticklabels=["Survived", "Died"], yticklabels=["Survived", "Died"])
        ax.set_title(name, fontweight="bold", color=c)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.suptitle("Confusion Matrices — Validation Set", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "confusion_matrices.png")


def fig_radar(y_val, models):
    rows = []
    for name, d in models.items():
        rows.append({
            "Model": name,
            "Recall": recall_score(y_val, d["pred"]),
            "F1-Score": f1_score(y_val, d["pred"]),
            "ROC-AUC": roc_auc_score(y_val, d["prob"]),
        })
    radar_df = pd.DataFrame(rows).set_index("Model")
    labels = list(radar_df.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (name, row) in enumerate(radar_df.iterrows()):
        vals = row.values.tolist() + [row.values[0]]
        ax.plot(angles, vals, "o-", lw=2, label=name, color=MODEL_COLORS[i])
        ax.fill(angles, vals, alpha=0.15, color=MODEL_COLORS[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Model Profiles — Validation", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    save_fig(fig, "model_radar.png")


def fig_per_model(y_val, models):
    for idx, (name, d) in enumerate(models.items()):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        metrics = {
            "Accuracy": accuracy_score(y_val, d["pred"]),
            "Precision": precision_score(y_val, d["pred"]),
            "Recall": recall_score(y_val, d["pred"]),
            "F1-Score": f1_score(y_val, d["pred"]),
            "ROC-AUC": roc_auc_score(y_val, d["prob"]),
        }
        axes[0].barh(list(metrics.keys()), list(metrics.values()), color=MODEL_COLORS[idx], edgecolor="white")
        axes[0].set_xlim(0, 1.05)
        axes[0].set_title(f"{name} — Metrics", fontweight="bold")
        cm = confusion_matrix(y_val, d["pred"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="PuBuGn", ax=axes[1],
                    xticklabels=["Survived", "Died"], yticklabels=["Survived", "Died"])
        axes[1].set_title("Confusion Matrix", fontweight="bold")
        plt.tight_layout()
        slug = name.lower().replace(" ", "_")
        save_fig(fig, f"model_{slug}.png")


def png_output(path: Path) -> dict:
    return {
        "output_type": "display_data",
        "data": {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")},
        "metadata": {},
    }


def enhance_notebook_markdown():
    if not NB.exists():
        return
    nb = json.loads(NB.read_text(encoding="utf-8"))
    nb["cells"][0]["source"] = [
        "# COVID-19 Mortality Prediction — Machine Learning Pipeline\n",
        "\n",
        "<p align=\"center\">\n",
        "  <img src=\"figures/eda_overview.png\" width=\"900\" alt=\"Dataset overview\"/>\n",
        "</p>\n",
        "\n",
        "End-to-end **binary classification**: predict patient **death vs survival** from clinical features (~1M records).\n",
        "\n",
        "| Step | What happens |\n",
        "|------|----------------|\n",
        "| 1–3 | Load, clean, train/val/test split (60/20/20) |\n",
        "| 4–5 | Class weights, train **LR**, **Random Forest**, **KNN** |\n",
        "| 6–8 | Compare models, ROC, radar, threshold tuning |\n",
        "| 9–10 | Feature importance, final test evaluation |\n",
        "\n",
        "> **Educational project only** — not for clinical decision-making.\n",
    ]
    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Updated notebook title cell")


def enhance_cleaning_title():
    clean = ROOT / "cleaning.ipynb"
    if not clean.exists():
        return
    nb = json.loads(clean.read_text(encoding="utf-8"))
    nb["cells"][0]["source"] = [
        "# COVID-19 Data Cleaning & EDA\n",
        "\n",
        "<p align=\"center\">\n",
        "  <img src=\"figures/eda_overview.png\" width=\"900\" alt=\"Dataset overview\"/>\n",
        "</p>\n",
        "\n",
        "Prepare raw Mexican COVID-19 surveillance data for modeling: handle encoding, missing values, and exploratory plots.\n",
        "\n",
        "**Next:** run [`modeling.ipynb`](modeling.ipynb) for training and evaluation.\n",
    ]
    clean.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Updated cleaning.ipynb title")


def embed_notebook():
    if not NB.exists():
        return
    nb = json.loads(NB.read_text(encoding="utf-8"))
    rules = [
        ("Target Class Distribution", "eda_overview.png"),
        ("'Logistic Regression', y_val", "model_logistic_regression.png"),
        ("'Random Forest', y_val", "model_random_forest.png"),
        ("'KNN', y_val", "model_knn.png"),
        ("Model Comparison — Validation Metrics", "model_comparison.png"),
        ("Confusion Matrices (Validation Set)", "confusion_matrices.png"),
        ("ROC Curves — Validation Set", "roc_curves.png"),
        ("subplot_kw=dict(polar=True)", "model_radar.png"),
        ("COVID-19 Mortality Prediction — Results Summary", "results_summary.png"),
    ]
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        for needle, fname in rules:
            if needle in src and (FIG / fname).exists():
                cell["outputs"] = [png_output(FIG / fname)]
                if cell.get("execution_count") is None:
                    cell["execution_count"] = 1
                break
    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Embedded figures into", NB.name)


def main():
    import generate_figures as gf
    df_full = gf.load_cleaned_df()
    gf.fig_pipeline()
    gf.fig_eda(df_full)
    gf.fig_model_comparison()
    gf.fig_results_summary()

    print("\nTraining sample models for evaluation plots...")
    df = load_cleaned_df(max_rows=120_000)
    y_val, models = train_all(df)
    fig_roc(y_val, models)
    fig_confusion(y_val, models)
    fig_radar(y_val, models)
    fig_per_model(y_val, models)
    enhance_notebook_markdown()
    enhance_cleaning_title()
    embed_notebook()


if __name__ == "__main__":
    main()
