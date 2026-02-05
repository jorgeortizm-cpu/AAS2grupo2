# -*- coding: utf-8 -*-
"""
1_ExploracionEDA.py - Analisis Exploratorio de Datos (EDA)

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan

Objetivo:
  Realizar el analisis exploratorio del dataset de empresas del Ecuador 2024,
  describir variables y clases, y generar visualizaciones guardadas en results/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.max_open_warning": 0})

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
print("  ANALISIS EXPLORATORIO DE DATOS (EDA)")
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

print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nColumnas: {list(df.columns)}")

# =========================================================
# 3. DESCRIPCION DE VARIABLES
# =========================================================
print("\n" + "=" * 60)
print("  3. DESCRIPCION DE VARIABLES")
print("=" * 60)

# 3.1 Tipos de datos
print("\n--- Tipos de datos ---")
print(df.dtypes)

# 3.2 Valores nulos
print("\n--- Valores nulos por columna ---")
nulos = df.isnull().sum()
print(nulos[nulos > 0] if nulos.sum() > 0 else "No hay valores nulos.")

# 3.3 Estadisticas descriptivas de variables numericas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print(f"\nVariables numericas ({len(num_cols)}): {num_cols}")
print(f"Variables categoricas ({len(cat_cols)}): {cat_cols}")

print("\n--- Estadisticas descriptivas (variables numericas) ---")
desc = df[num_cols].describe().T
desc["coef_var"] = (desc["std"] / desc["mean"]).abs()
print(desc.to_string())

# 3.4 Distribucion de variables categoricas
print("\n--- Distribucion de variables categoricas ---")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10).to_string())

# =========================================================
# 4. LIMPIEZA BASICA PARA VISUALIZACIONES
# =========================================================
# Convertir columnas numericas que puedan estar como object
for col in df.columns:
    if col not in cat_cols and col != "Año":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Recalcular columnas numericas despues de conversion
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Excluir Año de las numericas para analisis financiero
fin_cols = [c for c in num_cols if c != "Año"]

# =========================================================
# 5. CREACION DE VARIABLE OBJETIVO (Desempeno)
# =========================================================
print("\n" + "=" * 60)
print("  5. CREACION DE VARIABLE OBJETIVO")
print("=" * 60)

epsilon = 1e-7
df["Margen_Neto"] = df["UtilidadNeta"] / (df["IngresosTotales"] + epsilon)
df["ROA"] = df["UtilidadNeta"] / (df["Activo"] + epsilon)
df["ROE"] = df["UtilidadNeta"] / (df["Patrimonio"] + epsilon)

# Limpiar infinitos
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Margen_Neto"])

# Clasificar en 3 niveles
df["Desempeno"] = pd.qcut(
    df["Margen_Neto"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop"
)

print(f"\nDistribucion de la variable objetivo (Desempeno):")
print(df["Desempeno"].value_counts())
print(f"\nPorcentaje:")
print((df["Desempeno"].value_counts(normalize=True) * 100).round(2))

# =========================================================
# 6. VISUALIZACIONES
# =========================================================
print("\n" + "=" * 60)
print("  6. GENERACION DE VISUALIZACIONES")
print("=" * 60)

# Paleta de colores consistente
palette_desemp = {"Bajo": "#e74c3c", "Medio": "#f39c12", "Alto": "#27ae60"}
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# --- 6.1 Distribucion de la variable objetivo ---
print("\n[1/8] Distribucion de la variable objetivo...")
fig, ax = plt.subplots(figsize=(8, 5))
counts = df["Desempeno"].value_counts()
bars = ax.bar(counts.index, counts.values, color=[palette_desemp[x] for x in counts.index],
              edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{val:,}", ha="center", va="bottom", fontweight="bold")
ax.set_title("Distribucion de la Variable Objetivo: Desempeno Financiero", fontsize=14, fontweight="bold")
ax.set_xlabel("Nivel de Desempeno")
ax.set_ylabel("Cantidad de Empresas")
save_fig(fig, "01_distribucion_variable_objetivo.png")

# --- 6.2 Distribucion por Sector ---
print("[2/8] Distribucion por Sector...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sector_counts = df["Sector"].value_counts()
axes[0].bar(sector_counts.index, sector_counts.values, color=["#3498db", "#e67e22"],
            edgecolor="black", linewidth=0.5)
for i, (idx, val) in enumerate(sector_counts.items()):
    axes[0].text(i, val + 500, f"{val:,}", ha="center", fontweight="bold")
axes[0].set_title("Cantidad de Empresas por Sector", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Cantidad")

ct = pd.crosstab(df["Sector"], df["Desempeno"], normalize="index") * 100
ct[["Bajo", "Medio", "Alto"]].plot(kind="bar", stacked=True, ax=axes[1],
                                    color=[palette_desemp["Bajo"], palette_desemp["Medio"], palette_desemp["Alto"]],
                                    edgecolor="black", linewidth=0.5)
axes[1].set_title("Desempeno Financiero por Sector (%)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Sector")
axes[1].legend(title="Desempeno")
axes[1].tick_params(axis="x", rotation=0)
fig.tight_layout()
save_fig(fig, "02_distribucion_sector.png")

# --- 6.3 Histogramas de variables financieras ---
print("[3/8] Histogramas de variables financieras...")
plot_cols = ["Cant_Empleados", "Activo", "Patrimonio", "IngresoVentas",
             "UtilidadAntesImpuestos", "UtilidadEjercicio", "UtilidadNeta",
             "IR_Causado", "IngresosTotales"]
# Filtrar solo columnas que existan
plot_cols = [c for c in plot_cols if c in df.columns]

n = len(plot_cols)
ncols = 3
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes = axes.flatten()

for i, col in enumerate(plot_cols):
    data = df[col].dropna()
    # Usar escala log para mejor visualizacion si hay gran rango
    if data.max() > 1e6:
        log_data = np.log1p(data.clip(lower=0))
        axes[i].hist(log_data, bins=50, color="#3498db", edgecolor="black", linewidth=0.3, alpha=0.8)
        axes[i].set_xlabel(f"log(1 + {col})")
    else:
        axes[i].hist(data, bins=50, color="#3498db", edgecolor="black", linewidth=0.3, alpha=0.8)
        axes[i].set_xlabel(col)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Frecuencia")

# Ocultar ejes sobrantes
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Distribucion de Variables Financieras", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "03_histogramas_variables_financieras.png")

# --- 6.4 Boxplots por nivel de desempeno ---
print("[4/8] Boxplots por nivel de desempeno...")
indicadores = ["Margen_Neto", "ROA", "ROE"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, col in enumerate(indicadores):
    data = df[[col, "Desempeno"]].dropna()
    # Recortar outliers extremos para mejor visualizacion
    q01 = data[col].quantile(0.01)
    q99 = data[col].quantile(0.99)
    data_clip = data[(data[col] >= q01) & (data[col] <= q99)]

    sns.boxplot(data=data_clip, x="Desempeno", y=col, order=["Bajo", "Medio", "Alto"],
                palette=palette_desemp, ax=axes[i], fliersize=2)
    axes[i].set_title(f"{col} por Nivel de Desempeno", fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Desempeno")
    axes[i].set_ylabel(col)

fig.suptitle("Boxplots de Indicadores Financieros por Desempeno", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "04_boxplots_indicadores_desempeno.png")

# --- 6.5 Boxplots de variables financieras originales ---
print("[5/8] Boxplots de variables financieras originales...")
box_cols = ["Activo", "Patrimonio", "IngresoVentas", "UtilidadNeta", "IngresosTotales"]
box_cols = [c for c in box_cols if c in df.columns]

fig, axes = plt.subplots(1, len(box_cols), figsize=(4 * len(box_cols), 6))
if len(box_cols) == 1:
    axes = [axes]

for i, col in enumerate(box_cols):
    data = df[[col, "Desempeno"]].dropna()
    # Usar log para mejor visualizacion
    data["log_val"] = np.log1p(data[col].clip(lower=0))
    sns.boxplot(data=data, x="Desempeno", y="log_val", order=["Bajo", "Medio", "Alto"],
                palette=palette_desemp, ax=axes[i], fliersize=1)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_ylabel(f"log(1 + {col})")
    axes[i].set_xlabel("Desempeno")

fig.suptitle("Variables Financieras por Nivel de Desempeno (escala log)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "05_boxplots_variables_financieras.png")

# --- 6.6 Matriz de correlacion ---
print("[6/8] Matriz de correlacion...")
corr_cols = [c for c in fin_cols if c in df.columns] + ["Margen_Neto", "ROA", "ROE"]
corr_cols = list(dict.fromkeys(corr_cols))  # Eliminar duplicados manteniendo orden
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            annot_kws={"size": 8})
ax.set_title("Matriz de Correlacion - Variables Financieras e Indicadores", fontsize=14, fontweight="bold")
fig.tight_layout()
save_fig(fig, "06_matriz_correlacion.png")

# --- 6.7 Pairplot de indicadores clave ---
print("[7/8] Pairplot de indicadores clave (muestra)...")
sample_size = min(3000, len(df))
df_sample = df[["Margen_Neto", "ROA", "ROE", "Desempeno"]].dropna().sample(n=sample_size, random_state=42)

# Recortar extremos para visualizacion
for col in ["Margen_Neto", "ROA", "ROE"]:
    q01 = df_sample[col].quantile(0.01)
    q99 = df_sample[col].quantile(0.99)
    df_sample = df_sample[(df_sample[col] >= q01) & (df_sample[col] <= q99)]

g = sns.pairplot(df_sample, hue="Desempeno", palette=palette_desemp,
                 hue_order=["Bajo", "Medio", "Alto"],
                 diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
g.figure.suptitle("Relaciones entre Indicadores Financieros por Desempeno", fontsize=14, fontweight="bold", y=1.02)
save_fig(g.figure, "07_pairplot_indicadores.png")

# --- 6.8 Estadisticas descriptivas por clase ---
print("[8/8] Tabla de estadisticas por clase de desempeno...")
stats_cols = ["Cant_Empleados", "Activo", "Patrimonio", "IngresoVentas",
              "UtilidadNeta", "IngresosTotales", "Margen_Neto", "ROA", "ROE"]
stats_cols = [c for c in stats_cols if c in df.columns]

stats_by_class = df.groupby("Desempeno")[stats_cols].agg(["mean", "median", "std"]).round(4)

fig, ax = plt.subplots(figsize=(18, 8))
ax.axis("off")

# Crear tabla resumen simplificada
summary_data = []
for nivel in ["Bajo", "Medio", "Alto"]:
    row = [nivel]
    for col in stats_cols:
        mean_val = df[df["Desempeno"] == nivel][col].mean()
        if abs(mean_val) > 1000:
            row.append(f"{mean_val:,.0f}")
        else:
            row.append(f"{mean_val:.4f}")
    summary_data.append(row)

table = ax.table(
    cellText=summary_data,
    colLabels=["Desempeno"] + stats_cols,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.8)

# Colorear encabezados
for j in range(len(stats_cols) + 1):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colorear filas por nivel
row_colors = {"Bajo": "#fadbd8", "Medio": "#fdebd0", "Alto": "#d5f5e3"}
for i, nivel in enumerate(["Bajo", "Medio", "Alto"]):
    for j in range(len(stats_cols) + 1):
        table[i + 1, j].set_facecolor(row_colors[nivel])

ax.set_title("Estadisticas Descriptivas por Nivel de Desempeno (Media)", fontsize=14, fontweight="bold", pad=20)
save_fig(fig, "08_tabla_estadisticas_por_clase.png")

# =========================================================
# 7. RESUMEN FINAL
# =========================================================
print("\n" + "=" * 60)
print("  RESUMEN DEL EDA")
print("=" * 60)
print(f"  Total de registros analizados: {len(df):,}")
print(f"  Variables originales: {len(df.columns) - 4} + 4 derivadas")
print(f"  Sectores: {df['Sector'].nunique()}")
print(f"  Clases de desempeno: {df['Desempeno'].nunique()}")
print(f"\n  Imagenes generadas en: {RESULTS_DIR}")
print(f"  Total de graficos: 8")
print("=" * 60)
print("\n  Archivos generados:")
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith(".png"):
        print(f"    - {f}")
print("\nEDA completado exitosamente.")
