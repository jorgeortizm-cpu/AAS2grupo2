# -*- coding: utf-8 -*-
"""
3_EntrenarYEvaluar.py - Implementacion de Clasificadores

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan

Objetivo:
  Entrenar y evaluar 3 modelos de clasificacion:
  - Modelo 1: Arbol de Decision
  - Modelo 2: SVM (con ajuste de kernel y C)
  - Modelo 3: Random Forest
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURACION DE RUTAS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "DataSet2024.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def save_fig(fig, name):
    """Guarda una figura en la carpeta results."""
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Guardado: {path}")


# =========================================================
# 2. CARGA Y PREPROCESAMIENTO (replica del paso 2)
# =========================================================
print("=" * 60)
print("  IMPLEMENTACION DE CLASIFICADORES")
print("  Dataset: Empresas del Ecuador - 2024")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig", engine="python", on_bad_lines="skip")

# Normalizar nombres de columnas
df.columns = (
    df.columns.str.strip()
    .str.replace("\n", "_", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "", regex=False)
)

# Eliminar columna Año
col_ano = [c for c in df.columns if df[c].nunique() == 1 and df[c].dtype in ["int64", "float64"]]
if col_ano:
    df = df.drop(columns=col_ano)

# Convertir columnas numericas
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in df.columns:
    if col not in cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Crear variable objetivo
epsilon = 1e-7
df["Margen_Neto"] = df["UtilidadNeta"] / (df["IngresosTotales"] + epsilon)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Margen_Neto"])
df["Desempeno"] = pd.qcut(df["Margen_Neto"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop")

# Codificar categoricas
le_sector = LabelEncoder()
df["Sector"] = le_sector.fit_transform(df["Sector"].astype(str))

le_target = LabelEncoder()
df["Desempeno_cod"] = le_target.fit_transform(df["Desempeno"])
clases = list(le_target.classes_)

# Definir features y target
exclude = ["Desempeno", "Desempeno_cod", "Margen_Neto"]
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols]
y = df["Desempeno_cod"]

# Escalar
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

# Division 80/20 estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nDatos preparados:")
print(f"  Entrenamiento: {X_train.shape[0]:,} registros")
print(f"  Prueba:        {X_test.shape[0]:,} registros")
print(f"  Features:      {X_train.shape[1]}")
print(f"  Clases:        {clases}")

# =========================================================
# 3. MODELO 1: ARBOL DE DECISION
# =========================================================
print("\n" + "=" * 60)
print("  MODELO 1: ARBOL DE DECISION")
print("=" * 60)

t0 = time.time()
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)
dt_model.fit(X_train, y_train)
dt_time = time.time() - t0

y_pred_dt = dt_model.predict(X_test)

print(f"\n  Tiempo de entrenamiento: {dt_time:.2f}s")
print(f"  Profundidad del arbol:   {dt_model.get_depth()}")
print(f"  Hojas:                   {dt_model.get_n_leaves()}")
print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"  F1 (weighted): {f1_score(y_test, y_pred_dt, average='weighted'):.4f}")
print(f"\n  Reporte de Clasificacion:")
print(classification_report(y_test, y_pred_dt, target_names=clases))

# =========================================================
# 4. MODELO 2: SVM (con ajuste de kernel y C)
# =========================================================
print("=" * 60)
print("  MODELO 2: SVM (Support Vector Machine)")
print("  Ajuste de hiperparametros: kernel y C")
print("=" * 60)

# Usar muestra para GridSearch (SVM es costoso con 100k+ registros)
sample_size = min(15000, len(X_train))
idx_sample = np.random.RandomState(42).choice(X_train.index, size=sample_size, replace=False)
X_train_sample = X_train.loc[idx_sample]
y_train_sample = y_train.loc[idx_sample]

print(f"\n  Muestra para GridSearchCV: {sample_size:,} registros")
print(f"  Buscando mejores hiperparametros...")

param_grid_svm = {
    "kernel": ["rbf", "linear"],
    "C": [0.1, 1.0, 10.0],
}

svm_grid = GridSearchCV(
    SVC(class_weight="balanced", random_state=42),
    param_grid_svm,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=0
)

t0 = time.time()
svm_grid.fit(X_train_sample, y_train_sample)
svm_search_time = time.time() - t0

print(f"\n  GridSearch completado en {svm_search_time:.2f}s")
print(f"  Mejores hiperparametros: {svm_grid.best_params_}")
print(f"  Mejor F1 (CV): {svm_grid.best_score_:.4f}")

# Resultados del GridSearch
gs_results = pd.DataFrame(svm_grid.cv_results_)
gs_summary = gs_results[["param_kernel", "param_C", "mean_test_score", "std_test_score", "rank_test_score"]]
gs_summary = gs_summary.sort_values("rank_test_score")
print(f"\n  Resultados del GridSearch:")
print(gs_summary.to_string(index=False))

# Entrenar modelo final con mejores parametros sobre toda la muestra
print(f"\n  Entrenando SVM final con mejores parametros...")
t0 = time.time()
svm_model = SVC(
    kernel=svm_grid.best_params_["kernel"],
    C=svm_grid.best_params_["C"],
    class_weight="balanced",
    random_state=42,
    probability=True
)
svm_model.fit(X_train_sample, y_train_sample)
svm_time = time.time() - t0

y_pred_svm = svm_model.predict(X_test)

print(f"  Tiempo de entrenamiento: {svm_time:.2f}s")
print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"  F1 (weighted): {f1_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"\n  Reporte de Clasificacion:")
print(classification_report(y_test, y_pred_svm, target_names=clases))

# =========================================================
# 5. MODELO 3: RANDOM FOREST
# =========================================================
print("=" * 60)
print("  MODELO 3: RANDOM FOREST")
print("=" * 60)

t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_time = time.time() - t0

y_pred_rf = rf_model.predict(X_test)

print(f"\n  Tiempo de entrenamiento: {rf_time:.2f}s")
print(f"  Arboles:    {rf_model.n_estimators}")
print(f"  Max depth:  {rf_model.max_depth}")
print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  F1 (weighted): {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"\n  Reporte de Clasificacion:")
print(classification_report(y_test, y_pred_rf, target_names=clases))

# =========================================================
# 6. COMPARATIVA DE MODELOS
# =========================================================
print("=" * 60)
print("  COMPARATIVA DE MODELOS")
print("=" * 60)

modelos = {
    "Arbol de Decision": {"modelo": dt_model, "pred": y_pred_dt, "tiempo": dt_time},
    "SVM": {"modelo": svm_model, "pred": y_pred_svm, "tiempo": svm_time},
    "Random Forest": {"modelo": rf_model, "pred": y_pred_rf, "tiempo": rf_time},
}

comparativa = []
for nombre, info in modelos.items():
    pred = info["pred"]
    comparativa.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, pred),
        "F1 (weighted)": f1_score(y_test, pred, average="weighted"),
        "Precision (weighted)": precision_score(y_test, pred, average="weighted"),
        "Recall (weighted)": recall_score(y_test, pred, average="weighted"),
        "Tiempo (s)": info["tiempo"],
    })

df_comp = pd.DataFrame(comparativa)
print(f"\n{df_comp.to_string(index=False)}")

mejor = df_comp.loc[df_comp["F1 (weighted)"].idxmax()]
print(f"\n  MEJOR MODELO: {mejor['Modelo']} (F1 = {mejor['F1 (weighted)']:.4f})")

# =========================================================
# 7. VISUALIZACIONES
# =========================================================
print("\n" + "=" * 60)
print("  GENERACION DE VISUALIZACIONES")
print("=" * 60)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
palette_desemp = {"Bajo": "#e74c3c", "Medio": "#f39c12", "Alto": "#27ae60"}

# --- 7.1 Matrices de confusion ---
print("\n[1/5] Matrices de confusion...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, (nombre, info) in enumerate(modelos.items()):
    cm = confusion_matrix(y_test, info["pred"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=clases, yticklabels=clases, ax=axes[i],
                vmin=0, vmax=1, linewidths=0.5)
    acc = accuracy_score(y_test, info["pred"])
    f1 = f1_score(y_test, info["pred"], average="weighted")
    axes[i].set_title(f"{nombre}\nAcc={acc:.3f} | F1={f1:.3f}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Prediccion")
    axes[i].set_ylabel("Real")

fig.suptitle("Matrices de Confusion Normalizadas - Comparativa de Modelos",
             fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
save_fig(fig, "13_matrices_confusion_comparativa.png")

# --- 7.2 Comparativa de metricas ---
print("[2/5] Comparativa de metricas...")
fig, ax = plt.subplots(figsize=(12, 6))

metricas = ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)"]
x = np.arange(len(metricas))
width = 0.25
colors = ["#3498db", "#e67e22", "#27ae60"]

for i, (_, row) in enumerate(df_comp.iterrows()):
    valores = [row[m] for m in metricas]
    bars = ax.bar(x + i * width, valores, width, label=row["Modelo"],
                  color=colors[i], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Valor")
ax.set_title("Comparativa de Metricas entre Modelos", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metricas)
ax.legend(loc="lower right")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "14_comparativa_metricas.png")

# --- 7.3 Arbol de Decision (visualizacion) ---
print("[3/5] Visualizacion del Arbol de Decision (primeros niveles)...")
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(dt_model, max_depth=3, feature_names=feature_cols, class_names=clases,
          filled=True, rounded=True, fontsize=8, ax=ax,
          proportion=True, impurity=True)
ax.set_title("Arbol de Decision (primeros 3 niveles)", fontsize=16, fontweight="bold")
save_fig(fig, "15_arbol_decision_visualizacion.png")

# --- 7.4 Importancia de features (Random Forest) ---
print("[4/5] Importancia de features (Random Forest)...")
importancias = rf_model.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importancia": importancias
}).sort_values("Importancia", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = plt.cm.viridis(feat_imp["Importancia"] / feat_imp["Importancia"].max())
bars = ax.barh(feat_imp["Feature"], feat_imp["Importancia"], color=colors_bar, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, feat_imp["Importancia"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_title("Importancia de Features - Random Forest", fontsize=14, fontweight="bold")
ax.set_xlabel("Importancia (Gini)")
ax.grid(axis="x", linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "16_importancia_features_rf.png")

# --- 7.5 Resultados del GridSearch SVM ---
print("[5/5] Resultados del GridSearch SVM...")
fig, ax = plt.subplots(figsize=(10, 6))

for kernel in ["rbf", "linear"]:
    mask = gs_results["param_kernel"] == kernel
    subset = gs_results[mask].sort_values("param_C")
    ax.plot(subset["param_C"].astype(float), subset["mean_test_score"],
            marker="o", linewidth=2, markersize=8, label=f"kernel={kernel}")
    ax.fill_between(
        subset["param_C"].astype(float),
        subset["mean_test_score"] - subset["std_test_score"],
        subset["mean_test_score"] + subset["std_test_score"],
        alpha=0.15
    )

best_c = svm_grid.best_params_["C"]
best_k = svm_grid.best_params_["kernel"]
ax.axvline(x=best_c, color="red", linestyle="--", alpha=0.7, label=f"Mejor: C={best_c}, {best_k}")
ax.set_xscale("log")
ax.set_xlabel("Parametro C (escala log)", fontsize=12)
ax.set_ylabel("F1-Score (CV)", fontsize=12)
ax.set_title("GridSearchCV - SVM: Ajuste de Kernel y C", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "17_gridsearch_svm.png")

# =========================================================
# 8. EXPORTAR MODELOS A CARPETA Models/
# =========================================================
print("\n" + "=" * 60)
print("  EXPORTACION DE MODELOS")
print("=" * 60)

# Guardar los 3 modelos entrenados
modelos_export = {
    "arbol_decision": dt_model,
    "svm": svm_model,
    "random_forest": rf_model,
}

for nombre_archivo, modelo_obj in modelos_export.items():
    path_modelo = os.path.join(MODELS_DIR, f"{nombre_archivo}.pkl")
    joblib.dump(modelo_obj, path_modelo)
    print(f"  -> Modelo guardado: {path_modelo}")

# Guardar el scaler y los encoders para inferencia futura
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print(f"  -> Scaler guardado: {os.path.join(MODELS_DIR, 'scaler.pkl')}")

joblib.dump(le_target, os.path.join(MODELS_DIR, "label_encoder_target.pkl"))
print(f"  -> LabelEncoder target guardado: {os.path.join(MODELS_DIR, 'label_encoder_target.pkl')}")

joblib.dump(le_sector, os.path.join(MODELS_DIR, "label_encoder_sector.pkl"))
print(f"  -> LabelEncoder sector guardado: {os.path.join(MODELS_DIR, 'label_encoder_sector.pkl')}")

joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
print(f"  -> Feature columns guardado: {os.path.join(MODELS_DIR, 'feature_columns.pkl')}")

print(f"\n  Total de archivos exportados: {len(os.listdir(MODELS_DIR))}")
print(f"  Directorio: {MODELS_DIR}")

# =========================================================
# 9. TABLA RESUMEN FINAL
# =========================================================
print("\n" + "=" * 60)
print("  RESUMEN DE RESULTADOS")
print("=" * 60)
print(f"\n  {'Modelo':<22} {'Accuracy':>10} {'F1':>10} {'Tiempo':>10}")
print(f"  {'-'*52}")
for _, row in df_comp.iterrows():
    marcador = " <-- MEJOR" if row["Modelo"] == mejor["Modelo"] else ""
    print(f"  {row['Modelo']:<22} {row['Accuracy']:>10.4f} {row['F1 (weighted)']:>10.4f} {row['Tiempo (s)']:>9.2f}s{marcador}")

print(f"\n  SVM - Mejores hiperparametros: kernel={svm_grid.best_params_['kernel']}, C={svm_grid.best_params_['C']}")
print(f"  Random Forest - Feature mas importante: {feat_imp.iloc[-1]['Feature']} ({feat_imp.iloc[-1]['Importancia']:.4f})")
print(f"\n  Imagenes generadas en: {RESULTS_DIR}")
print("=" * 60)
print("\n  Archivos generados:")
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith(".png") and f.startswith("1") and int(f.split("_")[0]) >= 13:
        print(f"    - {f}")
print("\nEntrenamiento y evaluacion completados exitosamente.")
