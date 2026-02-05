# -*- coding: utf-8 -*-
"""
4_ComparacionExperimental.py - Comparacion Experimental de Modelos

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan

Objetivo:
  Comparar experimentalmente los 3 clasificadores entrenados mediante:
  - Precision, Recall, F1-Score (por clase y global)
  - Matrices de confusion
  - Tabla resumen de resultados
  - Visualizaciones comparativas (barplots y heatmaps)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURACION
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "DataSet2024.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Guardado: {path}")


# =========================================================
# 2. CARGA Y PREPROCESAMIENTO
# =========================================================
print("=" * 60)
print("  COMPARACION EXPERIMENTAL DE MODELOS")
print("  Dataset: Empresas del Ecuador - 2024")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig", engine="python", on_bad_lines="skip")
df.columns = (
    df.columns.str.strip()
    .str.replace("\n", "_", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "", regex=False)
)

# Eliminar Año
col_ano = [c for c in df.columns if df[c].nunique() == 1 and df[c].dtype in ["int64", "float64"]]
if col_ano:
    df = df.drop(columns=col_ano)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in df.columns:
    if col not in cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

epsilon = 1e-7
df["Margen_Neto"] = df["UtilidadNeta"] / (df["IngresosTotales"] + epsilon)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Margen_Neto"])
df["Desempeno"] = pd.qcut(df["Margen_Neto"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop")

le_sector = LabelEncoder()
df["Sector"] = le_sector.fit_transform(df["Sector"].astype(str))
le_target = LabelEncoder()
df["Desempeno_cod"] = le_target.fit_transform(df["Desempeno"])
clases = list(le_target.classes_)

exclude = ["Desempeno", "Desempeno_cod", "Margen_Neto"]
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols]
y = df["Desempeno_cod"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n  Entrenamiento: {X_train.shape[0]:,} | Prueba: {X_test.shape[0]:,} | Features: {X_train.shape[1]}")

# =========================================================
# 3. ENTRENAMIENTO DE LOS 3 MODELOS
# =========================================================
print("\n" + "=" * 60)
print("  ENTRENAMIENTO DE MODELOS")
print("=" * 60)

# Muestra para SVM
sample_size = min(15000, len(X_train))
idx_sample = np.random.RandomState(42).choice(X_train.index, size=sample_size, replace=False)
X_train_svm = X_train.loc[idx_sample]
y_train_svm = y_train.loc[idx_sample]

modelos_config = {
    "Arbol de Decision": DecisionTreeClassifier(
        max_depth=10, min_samples_split=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42
    ),
    "SVM (linear, C=10)": SVC(
        kernel="linear", C=10.0, class_weight="balanced",
        random_state=42, probability=True
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42
    ),
}

resultados = {}
for nombre, modelo in modelos_config.items():
    print(f"\n  Entrenando {nombre}...")
    t0 = time.time()
    if "SVM" in nombre:
        modelo.fit(X_train_svm, y_train_svm)
    else:
        modelo.fit(X_train, y_train)
    t_train = time.time() - t0

    y_pred = modelo.predict(X_test)

    resultados[nombre] = {
        "modelo": modelo,
        "y_pred": y_pred,
        "tiempo": t_train,
    }
    print(f"    Completado en {t_train:.2f}s")

# =========================================================
# 4. METRICAS DETALLADAS POR CLASE Y GLOBALES
# =========================================================
print("\n" + "=" * 60)
print("  METRICAS DETALLADAS POR MODELO Y POR CLASE")
print("=" * 60)

# 4.1 Recopilar metricas por clase para cada modelo
metricas_por_clase = {}
metricas_globales = []

for nombre, info in resultados.items():
    y_pred = info["y_pred"]

    print(f"\n  --- {nombre} ---")

    # Reporte por clase
    report = classification_report(y_test, y_pred, target_names=clases, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=clases))

    # Guardar metricas por clase
    for clase in clases:
        metricas_por_clase[f"{nombre} | {clase}"] = {
            "Modelo": nombre,
            "Clase": clase,
            "Precision": report[clase]["precision"],
            "Recall": report[clase]["recall"],
            "F1-Score": report[clase]["f1-score"],
            "Support": report[clase]["support"],
        }

    # Metricas globales
    metricas_globales.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted"),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score (weighted)": f1_score(y_test, y_pred, average="weighted"),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro"),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro"),
        "F1-Score (macro)": f1_score(y_test, y_pred, average="macro"),
        "Tiempo (s)": info["tiempo"],
    })

df_por_clase = pd.DataFrame(metricas_por_clase.values())
df_globales = pd.DataFrame(metricas_globales)

# =========================================================
# 5. TABLAS RESUMEN EN CONSOLA
# =========================================================
print("\n" + "=" * 60)
print("  TABLA RESUMEN - METRICAS GLOBALES (weighted)")
print("=" * 60)
print(df_globales[["Modelo", "Accuracy", "Precision (weighted)",
                    "Recall (weighted)", "F1-Score (weighted)", "Tiempo (s)"]].to_string(index=False))

print("\n" + "=" * 60)
print("  TABLA RESUMEN - METRICAS POR CLASE")
print("=" * 60)
print(df_por_clase[["Modelo", "Clase", "Precision", "Recall", "F1-Score", "Support"]].to_string(index=False))

# Mejor modelo
mejor_idx = df_globales["F1-Score (weighted)"].idxmax()
mejor = df_globales.loc[mejor_idx]
print(f"\n  MEJOR MODELO: {mejor['Modelo']} (F1 weighted = {mejor['F1-Score (weighted)']:.4f})")

# =========================================================
# 6. VISUALIZACIONES
# =========================================================
print("\n" + "=" * 60)
print("  GENERACION DE VISUALIZACIONES")
print("=" * 60)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.05)
colores_modelo = {
    "Arbol de Decision": "#3498db",
    "SVM (linear, C=10)": "#e67e22",
    "Random Forest": "#27ae60",
}
palette_clase = {"Alto": "#27ae60", "Bajo": "#e74c3c", "Medio": "#f39c12"}

# --- 6.1 Matrices de confusion detalladas (absoluta + normalizada) ---
print("\n[1/6] Matrices de confusion (absolutas y normalizadas)...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for i, (nombre, info) in enumerate(resultados.items()):
    cm = confusion_matrix(y_test, info["y_pred"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Fila 1: valores absolutos
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases, ax=axes[0, i],
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[0, i].set_title(f"{nombre}\n(Valores Absolutos)", fontsize=12, fontweight="bold")
    axes[0, i].set_xlabel("Prediccion")
    axes[0, i].set_ylabel("Real")

    # Fila 2: normalizadas
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="YlOrRd",
                xticklabels=clases, yticklabels=clases, ax=axes[1, i],
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[1, i].set_title(f"{nombre}\n(Normalizadas por clase real)", fontsize=12, fontweight="bold")
    axes[1, i].set_xlabel("Prediccion")
    axes[1, i].set_ylabel("Real")

fig.suptitle("Matrices de Confusion - Comparacion Experimental", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "18_matrices_confusion_detalladas.png")

# --- 6.2 Barplot comparativo de metricas globales ---
print("[2/6] Barplot de metricas globales...")
fig, ax = plt.subplots(figsize=(14, 7))

metricas_plot = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)"]
x = np.arange(len(metricas_plot))
width = 0.25
nombres_modelo = list(resultados.keys())

for i, nombre in enumerate(nombres_modelo):
    row = df_globales[df_globales["Modelo"] == nombre].iloc[0]
    valores = [row[m] for m in metricas_plot]
    bars = ax.bar(x + i * width, valores, width, label=nombre,
                  color=colores_modelo[nombre], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Valor de la Metrica", fontsize=12)
ax.set_title("Comparacion de Metricas Globales (weighted) por Modelo", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace(" (weighted)", "\n(weighted)") for m in metricas_plot], fontsize=10)
ax.legend(fontsize=11, loc="lower right")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
save_fig(fig, "19_barplot_metricas_globales.png")

# --- 6.3 Heatmap de metricas por clase y modelo ---
print("[3/6] Heatmap de metricas por clase...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, metrica in enumerate(["Precision", "Recall", "F1-Score"]):
    pivot = df_por_clase.pivot(index="Modelo", columns="Clase", values=metrica)
    pivot = pivot[clases]  # Ordenar columnas
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=axes[idx], cbar_kws={"shrink": 0.8})
    axes[idx].set_title(f"{metrica} por Clase", fontsize=13, fontweight="bold")
    axes[idx].set_ylabel("" if idx > 0 else "Modelo")
    axes[idx].set_xlabel("Clase")

fig.suptitle("Heatmap de Metricas por Clase y Modelo", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
save_fig(fig, "20_heatmap_metricas_por_clase.png")

# --- 6.4 Barplot de F1-Score por clase ---
print("[4/6] Barplot F1-Score por clase...")
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(clases))
width = 0.25

for i, nombre in enumerate(nombres_modelo):
    subset = df_por_clase[df_por_clase["Modelo"] == nombre]
    f1_vals = [subset[subset["Clase"] == c]["F1-Score"].values[0] for c in clases]
    bars = ax.bar(x + i * width, f1_vals, width, label=nombre,
                  color=colores_modelo[nombre], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("F1-Score por Clase y Modelo", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(clases, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
save_fig(fig, "21_barplot_f1_por_clase.png")

# --- 6.5 Radar chart comparativo ---
print("[5/6] Radar chart comparativo...")
metricas_radar = ["Accuracy", "Precision (weighted)", "Recall (weighted)",
                   "F1-Score (weighted)", "F1-Score (macro)"]
labels_radar = ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)",
                "F1-Score\n(weighted)", "F1-Score\n(macro)"]

angles = np.linspace(0, 2 * np.pi, len(metricas_radar), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for nombre in nombres_modelo:
    row = df_globales[df_globales["Modelo"] == nombre].iloc[0]
    valores = [row[m] for m in metricas_radar]
    valores += valores[:1]
    ax.plot(angles, valores, "o-", linewidth=2, markersize=6,
            label=nombre, color=colores_modelo[nombre])
    ax.fill(angles, valores, alpha=0.1, color=colores_modelo[nombre])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_radar, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.set_title("Perfil Comparativo de Modelos", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=10)
fig.tight_layout()
save_fig(fig, "22_radar_comparativo.png")

# --- 6.6 Tabla resumen visual ---
print("[6/6] Tabla resumen visual...")
fig, ax = plt.subplots(figsize=(18, 8))
ax.axis("off")

# Construir datos de la tabla
header = ["Modelo", "Accuracy", "Precision\n(weighted)", "Recall\n(weighted)",
          "F1-Score\n(weighted)", "Precision\n(macro)", "Recall\n(macro)",
          "F1-Score\n(macro)", "Tiempo (s)"]

cell_data = []
for _, row in df_globales.iterrows():
    cell_data.append([
        row["Modelo"],
        f"{row['Accuracy']:.4f}",
        f"{row['Precision (weighted)']:.4f}",
        f"{row['Recall (weighted)']:.4f}",
        f"{row['F1-Score (weighted)']:.4f}",
        f"{row['Precision (macro)']:.4f}",
        f"{row['Recall (macro)']:.4f}",
        f"{row['F1-Score (macro)']:.4f}",
        f"{row['Tiempo (s)']:.2f}",
    ])

table = ax.table(cellText=cell_data, colLabels=header, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Estilo del encabezado
for j in range(len(header)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Resaltar mejor modelo
for i in range(1, len(cell_data) + 1):
    modelo_nombre = cell_data[i - 1][0]
    if modelo_nombre == mejor["Modelo"]:
        for j in range(len(header)):
            table[i, j].set_facecolor("#d5f5e3")
            table[i, j].set_text_props(fontweight="bold")
    else:
        for j in range(len(header)):
            table[i, j].set_facecolor("#eaf2f8" if i % 2 == 0 else "#ffffff")

ax.set_title("Tabla Resumen - Comparacion Experimental de Clasificadores\n"
             f"(Mejor modelo resaltado en verde: {mejor['Modelo']})",
             fontsize=14, fontweight="bold", pad=25)
save_fig(fig, "23_tabla_resumen_comparacion.png")

# =========================================================
# 7. TABLA DE METRICAS POR CLASE (visual)
# =========================================================
print("\n  Generando tabla de metricas por clase...")
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis("off")

header_clase = ["Modelo", "Clase", "Precision", "Recall", "F1-Score", "Support"]
cell_clase = []
for _, row in df_por_clase.iterrows():
    cell_clase.append([
        row["Modelo"], row["Clase"],
        f"{row['Precision']:.4f}", f"{row['Recall']:.4f}",
        f"{row['F1-Score']:.4f}", f"{int(row['Support']):,}"
    ])

table2 = ax.table(cellText=cell_clase, colLabels=header_clase, cellLoc="center", loc="center")
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 2)

for j in range(len(header_clase)):
    table2[0, j].set_facecolor("#2c3e50")
    table2[0, j].set_text_props(color="white", fontweight="bold")

# Colorear por modelo
color_filas = {
    "Arbol de Decision": "#d6eaf8",
    "SVM (linear, C=10)": "#fdebd0",
    "Random Forest": "#d5f5e3",
}
for i in range(1, len(cell_clase) + 1):
    modelo_nombre = cell_clase[i - 1][0]
    bg = color_filas.get(modelo_nombre, "#ffffff")
    for j in range(len(header_clase)):
        table2[i, j].set_facecolor(bg)

ax.set_title("Metricas Detalladas por Clase y Modelo", fontsize=14, fontweight="bold", pad=25)
save_fig(fig, "24_tabla_metricas_por_clase.png")

# =========================================================
# 8. RESUMEN FINAL
# =========================================================
print("\n" + "=" * 60)
print("  RESUMEN DE LA COMPARACION EXPERIMENTAL")
print("=" * 60)
print(f"\n  {'Modelo':<22} {'Accuracy':>10} {'F1-w':>8} {'F1-m':>8} {'Prec-w':>8} {'Rec-w':>8}")
print(f"  {'-' * 66}")
for _, row in df_globales.iterrows():
    marca = " <--" if row["Modelo"] == mejor["Modelo"] else ""
    print(f"  {row['Modelo']:<22} {row['Accuracy']:>10.4f} {row['F1-Score (weighted)']:>8.4f} "
          f"{row['F1-Score (macro)']:>8.4f} {row['Precision (weighted)']:>8.4f} "
          f"{row['Recall (weighted)']:>8.4f}{marca}")

print(f"\n  MEJOR MODELO: {mejor['Modelo']}")
print(f"    F1-Score (weighted): {mejor['F1-Score (weighted)']:.4f}")
print(f"    F1-Score (macro):    {mejor['F1-Score (macro)']:.4f}")
print(f"\n  Imagenes generadas en: {RESULTS_DIR}")
print("=" * 60)
print("\n  Archivos generados:")
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith(".png") and f[:2].isdigit() and int(f.split("_")[0]) >= 18:
        print(f"    - {f}")
print("\nComparacion experimental completada exitosamente.")
