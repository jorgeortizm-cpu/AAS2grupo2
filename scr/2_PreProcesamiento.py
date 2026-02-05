# -*- coding: utf-8 -*-
"""
2_PreProcesamiento.py - Preprocesamiento de Datos

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan

Objetivo:
  Preprocesar el dataset de empresas del Ecuador 2024:
  - Eliminar columna Ano (no aporta al modelo)
  - Tratamiento de nulos
  - Codificacion de variables categoricas
  - Escalado de variables numericas
  - Division en conjunto de entrenamiento y prueba (80/20)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURACION DE RUTAS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "DataSet2024.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(fig, name):
    """Guarda una figura en la carpeta results."""
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Guardado: {path}")


# =========================================================
# 2. CARGA DEL DATASET
# =========================================================
print("=" * 60)
print("  PREPROCESAMIENTO DE DATOS")
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

print(f"\nDataset original: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")

# =========================================================
# 3. ELIMINACION DE COLUMNA AÑO
# =========================================================
print("\n" + "=" * 60)
print("  3. ELIMINACION DE COLUMNA AÑO")
print("=" * 60)

# La columna Año tiene un unico valor (2024) y no aporta poder predictivo
col_ano = [c for c in df.columns if "o" in c.lower() and df[c].nunique() == 1]
if not col_ano:
    # Buscar por posicion (primera columna es Año)
    col_ano = [df.columns[0]]

print(f"Columna identificada: '{col_ano[0]}' (valor unico: {df[col_ano[0]].unique()})")
df = df.drop(columns=col_ano)
print(f"Dataset tras eliminar Año: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Columnas restantes: {list(df.columns)}")

# =========================================================
# 4. TRATAMIENTO DE VALORES NULOS
# =========================================================
print("\n" + "=" * 60)
print("  4. TRATAMIENTO DE VALORES NULOS")
print("=" * 60)

# 4.1 Diagnostico de nulos
nulos_antes = df.isnull().sum()
total_nulos = nulos_antes.sum()
print(f"\nTotal de valores nulos: {total_nulos}")

if total_nulos > 0:
    print("\nNulos por columna:")
    print(nulos_antes[nulos_antes > 0])
    pct_nulos = (nulos_antes / len(df) * 100).round(2)
    print("\nPorcentaje de nulos:")
    print(pct_nulos[pct_nulos > 0])
else:
    print("No se encontraron valores nulos en el dataset.")

# 4.2 Convertir columnas numericas
num_cols_orig = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Forzar conversion numerica en columnas que deban serlo
for col in df.columns:
    if col not in cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Recalcular despues de conversion
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 4.3 Verificar nulos post-conversion
nulos_post = df.isnull().sum()
total_nulos_post = nulos_post.sum()
if total_nulos_post > 0:
    print(f"\nNulos tras conversion de tipos: {total_nulos_post}")
    print(nulos_post[nulos_post > 0])
    # Rellenar nulos numericos con la mediana (robusto ante outliers)
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(f"  {col}: {nulos_post[col]} nulos rellenados con mediana ({mediana:,.2f})")
    # Rellenar nulos categoricos con la moda
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            moda = df[col].mode()[0]
            df[col] = df[col].fillna(moda)
            print(f"  {col}: nulos rellenados con moda ('{moda}')")

print(f"\nNulos finales: {df.isnull().sum().sum()}")

# =========================================================
# 5. CREACION DE VARIABLE OBJETIVO (Desempeno)
# =========================================================
print("\n" + "=" * 60)
print("  5. CREACION DE VARIABLE OBJETIVO")
print("=" * 60)

epsilon = 1e-7
df["Margen_Neto"] = df["UtilidadNeta"] / (df["IngresosTotales"] + epsilon)

# Limpiar infinitos
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Margen_Neto"])

# Clasificar en 3 niveles
df["Desempeno"] = pd.qcut(
    df["Margen_Neto"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop"
)

print(f"Registros con variable objetivo: {len(df):,}")
print(f"\nDistribucion:")
print(df["Desempeno"].value_counts())

# =========================================================
# 6. CODIFICACION DE VARIABLES CATEGORICAS
# =========================================================
print("\n" + "=" * 60)
print("  6. CODIFICACION DE VARIABLES CATEGORICAS")
print("=" * 60)

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
# Excluir la variable objetivo si esta como categorica
cat_cols = [c for c in cat_cols if c != "Desempeno"]

print(f"Variables categoricas a codificar: {cat_cols}")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    print(f"\n  {col}:")
    print(f"    Valores originales: {df[col].unique()}")
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"    Codificacion: {mapping}")

# Codificar variable objetivo
le_target = LabelEncoder()
df["Desempeno_cod"] = le_target.fit_transform(df["Desempeno"])
mapping_target = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print(f"\n  Variable objetivo (Desempeno):")
print(f"    Codificacion: {mapping_target}")

# =========================================================
# 7. ESCALADO DE VARIABLES NUMERICAS
# =========================================================
print("\n" + "=" * 60)
print("  7. ESCALADO DE VARIABLES NUMERICAS (StandardScaler)")
print("=" * 60)

# Definir features (excluir variable objetivo y Margen_Neto usado para crear target)
exclude = ["Desempeno", "Desempeno_cod", "Margen_Neto"]
feature_cols = [c for c in df.columns if c not in exclude]

print(f"Features para el modelo ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].copy()
y = df["Desempeno_cod"].copy()

# Estadisticas antes del escalado
print("\n--- Estadisticas ANTES del escalado ---")
stats_antes = X.describe().T[["mean", "std", "min", "max"]]
print(stats_antes.to_string())

# Aplicar StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

# Estadisticas despues del escalado
print("\n--- Estadisticas DESPUES del escalado ---")
stats_despues = X_scaled.describe().T[["mean", "std", "min", "max"]]
print(stats_despues.round(4).to_string())

# =========================================================
# 8. DIVISION ENTRENAMIENTO / PRUEBA (80/20)
# =========================================================
print("\n" + "=" * 60)
print("  8. DIVISION ENTRENAMIENTO / PRUEBA (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n  Dataset total:       {X_scaled.shape[0]:,} registros")
print(f"  Entrenamiento (80%): {X_train.shape[0]:,} registros")
print(f"  Prueba (20%):        {X_test.shape[0]:,} registros")
print(f"  Features:            {X_train.shape[1]}")

# Verificar estratificacion
print("\n--- Distribucion de clases (estratificada) ---")
dist_train = y_train.value_counts(normalize=True).sort_index() * 100
dist_test = y_test.value_counts(normalize=True).sort_index() * 100
dist_df = pd.DataFrame({
    "Entrenamiento (%)": dist_train.round(2),
    "Prueba (%)": dist_test.round(2)
})
dist_df.index = [le_target.inverse_transform([i])[0] for i in dist_df.index]
print(dist_df.to_string())

# =========================================================
# 9. VISUALIZACIONES DEL PREPROCESAMIENTO
# =========================================================
print("\n" + "=" * 60)
print("  9. GENERACION DE VISUALIZACIONES")
print("=" * 60)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
palette_desemp = {"Bajo": "#e74c3c", "Medio": "#f39c12", "Alto": "#27ae60"}

# --- 9.1 Comparativa antes/despues del escalado ---
print("\n[1/4] Comparativa antes/despues del escalado...")
sample_cols = feature_cols[:6]  # Primeras 6 features para visualizacion
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Antes
X[sample_cols].boxplot(ax=axes[0], patch_artist=True,
                        boxprops=dict(facecolor="#3498db", alpha=0.6),
                        medianprops=dict(color="red", linewidth=2))
axes[0].set_title("ANTES del Escalado (valores originales)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Valor")
axes[0].tick_params(axis="x", rotation=30)

# Despues
X_scaled[sample_cols].boxplot(ax=axes[1], patch_artist=True,
                               boxprops=dict(facecolor="#27ae60", alpha=0.6),
                               medianprops=dict(color="red", linewidth=2))
axes[1].set_title("DESPUES del Escalado (StandardScaler)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Valor escalado")
axes[1].tick_params(axis="x", rotation=30)

fig.suptitle("Efecto del Escalado en Variables Numericas", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "09_comparativa_escalado.png")

# --- 9.2 Distribucion train/test por clase ---
print("[2/4] Distribucion train/test por clase...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

clases = le_target.classes_
colors = [palette_desemp[c] for c in clases]

# Train
train_counts = y_train.value_counts().sort_index()
train_labels = [le_target.inverse_transform([i])[0] for i in train_counts.index]
bars = axes[0].bar(train_labels, train_counts.values, color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, train_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=10)
axes[0].set_title(f"Entrenamiento ({X_train.shape[0]:,} registros)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Desempeno")

# Test
test_counts = y_test.value_counts().sort_index()
test_labels = [le_target.inverse_transform([i])[0] for i in test_counts.index]
bars = axes[1].bar(test_labels, test_counts.values, color=colors, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, test_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=10)
axes[1].set_title(f"Prueba ({X_test.shape[0]:,} registros)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Desempeno")

fig.suptitle("Division Estratificada: Entrenamiento (80%) vs Prueba (20%)",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "10_distribucion_train_test.png")

# --- 9.3 Distribucion de features escaladas ---
print("[3/4] Distribucion de features escaladas...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(feature_cols[:10]):
    axes[i].hist(X_scaled[col], bins=50, color="#3498db", edgecolor="black",
                 linewidth=0.3, alpha=0.8)
    axes[i].axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_ylabel("Frecuencia")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribucion de Features tras StandardScaler", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "11_distribucion_features_escaladas.png")

# --- 9.4 Resumen del preprocesamiento (tabla) ---
print("[4/4] Tabla resumen del preprocesamiento...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

resumen_data = [
    ["Registros originales", f"{134865:,}"],
    ["Columna eliminada", "Año (valor unico 2024, sin poder predictivo)"],
    ["Valores nulos", f"{total_nulos} (tratados con mediana/moda)"],
    ["Variable categorica codificada", f"Sector -> LabelEncoder ({len(encoders)} variable)"],
    ["Variable objetivo", "Desempeno (Bajo=0, Medio=2, Alto=1)"],
    ["Metodo de escalado", "StandardScaler (media=0, std=1)"],
    ["Features finales", f"{len(feature_cols)} variables"],
    ["Division", "80% entrenamiento / 20% prueba (estratificada)"],
    ["Registros entrenamiento", f"{X_train.shape[0]:,}"],
    ["Registros prueba", f"{X_test.shape[0]:,}"],
]

table = ax.table(
    cellText=resumen_data,
    colLabels=["Concepto", "Detalle"],
    cellLoc="left",
    loc="center",
    colWidths=[0.35, 0.65],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for j in range(2):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")

for i in range(1, len(resumen_data) + 1):
    color = "#eaf2f8" if i % 2 == 0 else "#ffffff"
    for j in range(2):
        table[i, j].set_facecolor(color)

ax.set_title("Resumen del Preprocesamiento de Datos", fontsize=14, fontweight="bold", pad=20)
save_fig(fig, "12_resumen_preprocesamiento.png")

# =========================================================
# 10. RESUMEN FINAL
# =========================================================
print("\n" + "=" * 60)
print("  RESUMEN DEL PREPROCESAMIENTO")
print("=" * 60)
print(f"  Columna eliminada: Año")
print(f"  Nulos tratados: {total_nulos}")
print(f"  Variables categoricas codificadas: {len(encoders)} (LabelEncoder)")
print(f"  Escalado: StandardScaler sobre {len(feature_cols)} features")
print(f"  Entrenamiento: {X_train.shape[0]:,} registros (80%)")
print(f"  Prueba: {X_test.shape[0]:,} registros (20%)")
print(f"\n  Imagenes generadas en: {RESULTS_DIR}")
print("=" * 60)
print("\n  Archivos generados:")
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith(".png") and int(f.split("_")[0]) >= 9:
        print(f"    - {f}")
print("\nPreprocesamiento completado exitosamente.")
