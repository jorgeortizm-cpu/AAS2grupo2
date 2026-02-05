# -*- coding: utf-8 -*-
"""
5_ProbarModelosProduccion.py - Prueba de Modelos en Produccion

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan

Objetivo:
  Cargar los modelos exportados (.pkl) y probarlos con datos nuevos
  provenientes de un archivo JSON, generando un informe de eficiencia
  y confiabilidad por cada modelo.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. CONFIGURACION DE RUTAS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_PATH = os.path.join(BASE_DIR, "Data", "datos_prueba_produccion.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Guardado: {path}")


# =========================================================
# 2. CARGA DE MODELOS Y ARTEFACTOS
# =========================================================
print("=" * 60)
print("  PRUEBA DE MODELOS EN PRODUCCION")
print("=" * 60)

print("\n  Cargando modelos y artefactos...")

modelos = {
    "Arbol de Decision": joblib.load(os.path.join(MODELS_DIR, "arbol_decision.pkl")),
    "SVM": joblib.load(os.path.join(MODELS_DIR, "svm.pkl")),
    "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")),
}
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le_target = joblib.load(os.path.join(MODELS_DIR, "label_encoder_target.pkl"))
le_sector = joblib.load(os.path.join(MODELS_DIR, "label_encoder_sector.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

print(f"  Modelos cargados: {list(modelos.keys())}")
print(f"  Features esperadas: {feature_cols}")
print(f"  Clases: {list(le_target.classes_)}")

# =========================================================
# 3. CARGA DE DATOS DE PRUEBA (JSON)
# =========================================================
print("\n" + "=" * 60)
print("  CARGA DE DATOS DE PRUEBA")
print("=" * 60)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data_json = json.load(f)

empresas = data_json["empresas"]
print(f"\n  Archivo: {DATA_PATH}")
print(f"  Empresas de prueba: {len(empresas)}")

# Mostrar resumen de empresas
print(f"\n  {'ID':<4} {'Empresa':<38} {'Sector':<20} {'Esperado':<10}")
print(f"  {'-' * 72}")
for emp in empresas:
    print(f"  {emp['id']:<4} {emp['nombre']:<38} {emp['Sector']:<20} {emp['desempeno_esperado']:<10}")

# =========================================================
# 4. PREPARAR DATOS PARA INFERENCIA
# =========================================================
print("\n" + "=" * 60)
print("  PREPARACION DE DATOS")
print("=" * 60)

# Extraer features y etiquetas esperadas
df_test = pd.DataFrame(empresas)
nombres_empresas = df_test["nombre"].tolist()
ids_empresas = df_test["id"].tolist()
esperados = df_test["desempeno_esperado"].tolist()

# Codificar Sector
df_test["Sector"] = le_sector.transform(df_test["Sector"].astype(str))

# Seleccionar features en el orden correcto
X_prod = df_test[feature_cols].copy()

# Escalar
X_prod_scaled = pd.DataFrame(
    scaler.transform(X_prod), columns=feature_cols, index=X_prod.index
)

# Codificar etiquetas esperadas
esperados_cod = le_target.transform(esperados)

print(f"  Features preparadas: {X_prod_scaled.shape}")
print(f"  Datos escalados correctamente.")

# =========================================================
# 5. INFERENCIA CON CADA MODELO
# =========================================================
print("\n" + "=" * 60)
print("  INFERENCIA - PREDICCIONES POR MODELO")
print("=" * 60)

resultados_por_modelo = {}

for nombre_modelo, modelo in modelos.items():
    print(f"\n  --- {nombre_modelo} ---")

    # Prediccion
    y_pred_cod = modelo.predict(X_prod_scaled)
    y_pred_labels = le_target.inverse_transform(y_pred_cod)

    # Probabilidades (confianza)
    if hasattr(modelo, "predict_proba"):
        probas = modelo.predict_proba(X_prod_scaled)
        confianzas = np.max(probas, axis=1)
    else:
        confianzas = np.ones(len(y_pred_cod))  # SVM sin proba

    # Aciertos
    aciertos = [pred == esp for pred, esp in zip(y_pred_labels, esperados)]

    # Guardar resultados
    resultados_por_modelo[nombre_modelo] = {
        "predicciones": y_pred_labels,
        "confianzas": confianzas,
        "aciertos": aciertos,
    }

    # Imprimir detalle
    print(f"  {'Empresa':<38} {'Esperado':<10} {'Prediccion':<12} {'Confianza':>10} {'Resultado':>10}")
    print(f"  {'-' * 80}")
    for i in range(len(empresas)):
        estado = "OK" if aciertos[i] else "FALLO"
        print(f"  {nombres_empresas[i]:<38} {esperados[i]:<10} {y_pred_labels[i]:<12} "
              f"{confianzas[i]:>9.1%} {estado:>10}")

    total_aciertos = sum(aciertos)
    accuracy = total_aciertos / len(aciertos)
    confianza_media = np.mean(confianzas)
    print(f"\n  Aciertos: {total_aciertos}/{len(aciertos)} | Accuracy: {accuracy:.1%} | Confianza media: {confianza_media:.1%}")

# =========================================================
# 6. INFORME DE EFICIENCIA Y CONFIABILIDAD
# =========================================================
print("\n" + "=" * 60)
print("  INFORME DE EFICIENCIA Y CONFIABILIDAD")
print("=" * 60)

# 6.1 Tabla resumen por modelo
informe_modelos = []
for nombre_modelo, res in resultados_por_modelo.items():
    aciertos = res["aciertos"]
    confianzas = res["confianzas"]
    n = len(aciertos)
    n_ok = sum(aciertos)

    # Desglose por clase
    por_clase = {}
    for clase in le_target.classes_:
        idx_clase = [i for i, e in enumerate(esperados) if e == clase]
        if idx_clase:
            aciertos_clase = sum(res["aciertos"][i] for i in idx_clase)
            conf_clase = np.mean([res["confianzas"][i] for i in idx_clase])
            por_clase[clase] = {
                "total": len(idx_clase),
                "aciertos": aciertos_clase,
                "accuracy": aciertos_clase / len(idx_clase),
                "confianza_media": conf_clase,
            }

    informe_modelos.append({
        "Modelo": nombre_modelo,
        "Empresas": n,
        "Aciertos": n_ok,
        "Fallos": n - n_ok,
        "Accuracy (%)": round(n_ok / n * 100, 2),
        "Confianza Media (%)": round(np.mean(confianzas) * 100, 2),
        "Confianza Min (%)": round(np.min(confianzas) * 100, 2),
        "Confianza Max (%)": round(np.max(confianzas) * 100, 2),
        "Detalle por clase": por_clase,
    })

df_informe = pd.DataFrame(informe_modelos)
print("\n  TABLA RESUMEN:")
print(df_informe[["Modelo", "Empresas", "Aciertos", "Fallos",
                   "Accuracy (%)", "Confianza Media (%)",
                   "Confianza Min (%)", "Confianza Max (%)"]].to_string(index=False))

# 6.2 Detalle por clase
print("\n  DESGLOSE POR CLASE:")
for info in informe_modelos:
    print(f"\n  {info['Modelo']}:")
    print(f"    {'Clase':<10} {'Total':>6} {'Aciertos':>9} {'Accuracy':>10} {'Confianza':>11}")
    print(f"    {'-' * 46}")
    for clase, det in info["Detalle por clase"].items():
        print(f"    {clase:<10} {det['total']:>6} {det['aciertos']:>9} "
              f"{det['accuracy']:>9.1%} {det['confianza_media']:>10.1%}")

# =========================================================
# 7. DETALLE DE PREDICCIONES POR EMPRESA
# =========================================================
print("\n" + "=" * 60)
print("  DETALLE COMPLETO POR EMPRESA")
print("=" * 60)

# Construir tabla cruzada empresa x modelo
detalle_empresas = []
for i in range(len(empresas)):
    fila = {
        "ID": ids_empresas[i],
        "Empresa": nombres_empresas[i],
        "Esperado": esperados[i],
    }
    coincidencias = 0
    for nombre_modelo, res in resultados_por_modelo.items():
        pred = res["predicciones"][i]
        conf = res["confianzas"][i]
        ok = res["aciertos"][i]
        fila[f"{nombre_modelo} (pred)"] = pred
        fila[f"{nombre_modelo} (conf)"] = f"{conf:.1%}"
        fila[f"{nombre_modelo} (ok)"] = "OK" if ok else "FALLO"
        if ok:
            coincidencias += 1
    fila["Consenso"] = f"{coincidencias}/{len(modelos)}"
    detalle_empresas.append(fila)

df_detalle = pd.DataFrame(detalle_empresas)
print(f"\n{df_detalle.to_string(index=False)}")

# =========================================================
# 8. VISUALIZACIONES
# =========================================================
print("\n" + "=" * 60)
print("  GENERACION DE VISUALIZACIONES")
print("=" * 60)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.05)
colores_modelo = {
    "Arbol de Decision": "#3498db",
    "SVM": "#e67e22",
    "Random Forest": "#27ae60",
}

# --- 8.1 Barplot de Accuracy por modelo ---
print("\n[1/5] Accuracy por modelo...")
fig, ax = plt.subplots(figsize=(10, 6))
nombres = list(resultados_por_modelo.keys())
accuracies = [sum(r["aciertos"]) / len(r["aciertos"]) * 100 for r in resultados_por_modelo.values()]
colores = [colores_modelo[n] for n in nombres]

bars = ax.bar(nombres, accuracies, color=colores, edgecolor="black", linewidth=0.5, width=0.5)
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy en Datos de Produccion (10 empresas de prueba)", fontsize=14, fontweight="bold")
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", alpha=0.4)
save_fig(fig, "25_accuracy_produccion.png")

# --- 8.2 Radar de confiabilidad por modelo ---
print("[2/7] Radar de confiabilidad por modelo...")

# Calcular metricas de confiabilidad para cada modelo
radar_metrics = {}
for nombre_modelo, res in resultados_por_modelo.items():
    confs = res["confianzas"]
    aciertos = res["aciertos"]
    n = len(aciertos)

    # Accuracy por clase
    acc_por_clase = {}
    for clase in le_target.classes_:
        idx_c = [i for i, e in enumerate(esperados) if e == clase]
        if idx_c:
            acc_por_clase[clase] = sum(aciertos[i] for i in idx_c) / len(idx_c) * 100

    radar_metrics[nombre_modelo] = {
        "Accuracy\nGeneral": sum(aciertos) / n * 100,
        "Confianza\nMedia": np.mean(confs) * 100,
        "Confianza\nMinima": np.min(confs) * 100,
        "Acc. Clase\nAlto": acc_por_clase.get("Alto", 0),
        "Acc. Clase\nBajo": acc_por_clase.get("Bajo", 0),
        "Acc. Clase\nMedio": acc_por_clase.get("Medio", 0),
    }

metric_names = list(list(radar_metrics.values())[0].keys())
n_metrics = len(metric_names)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for nombre_modelo in modelos.keys():
    valores = [radar_metrics[nombre_modelo][m] for m in metric_names]
    valores += valores[:1]
    ax.plot(angles, valores, "o-", linewidth=2.5, markersize=8,
            label=nombre_modelo, color=colores_modelo[nombre_modelo])
    ax.fill(angles, valores, alpha=0.12, color=colores_modelo[nombre_modelo])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names, fontsize=10, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="gray")
ax.set_title("Radar de Confiabilidad por Modelo\n(Accuracy, Confianza y Desempeno por Clase)",
             fontsize=14, fontweight="bold", pad=25)
ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.02), fontsize=11,
          frameon=True, fancybox=True, shadow=True)
save_fig(fig, "26_radar_confiabilidad_modelos.png")

# --- 8.3 Radar de confianza por empresa (mejor modelo) ---
print("[3/7] Radar de confianza por empresa (todos los modelos)...")

fig, axes = plt.subplots(2, 5, figsize=(24, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()

empresa_angles = np.linspace(0, 2 * np.pi, len(modelos), endpoint=False).tolist()
empresa_angles += empresa_angles[:1]
modelo_labels = [n.replace(" ", "\n") for n in modelos.keys()]

for i, emp in enumerate(empresas):
    ax = axes[i]
    valores = []
    colores_emp = []
    for nombre_modelo in modelos.keys():
        conf = resultados_por_modelo[nombre_modelo]["confianzas"][i] * 100
        valores.append(conf)
        colores_emp.append(colores_modelo[nombre_modelo])

    valores_plot = valores + valores[:1]
    ax.plot(empresa_angles, valores_plot, "o-", linewidth=2, markersize=6, color="#2c3e50")
    ax.fill(empresa_angles, valores_plot, alpha=0.2, color="#3498db")

    # Colorear puntos segun acierto/fallo
    for j, nombre_modelo in enumerate(modelos.keys()):
        ok = resultados_por_modelo[nombre_modelo]["aciertos"][i]
        color_punto = "#27ae60" if ok else "#e74c3c"
        ax.plot(empresa_angles[j], valores[j], "o", markersize=10, color=color_punto,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    ax.set_xticks(empresa_angles[:-1])
    ax.set_xticklabels(modelo_labels, fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color="gray")

    esperado_color = {"Alto": "#27ae60", "Medio": "#f39c12", "Bajo": "#e74c3c"}
    ax.set_title(f"{emp['id']}. {emp['nombre'][:20]}\nEsperado: {esperados[i]}",
                 fontsize=9, fontweight="bold", color=esperado_color[esperados[i]], pad=12)

fig.suptitle("Radar de Confianza por Empresa\n(Verde=Acierto, Rojo=Fallo en cada modelo)",
             fontsize=15, fontweight="bold", y=1.04)
fig.tight_layout()
save_fig(fig, "27_radar_confianza_por_empresa.png")

# --- 8.4 Heatmap de predicciones vs esperado ---
print("[4/7] Heatmap de predicciones por empresa...")
fig, ax = plt.subplots(figsize=(14, 8))

# Construir matriz: filas=empresas, cols=modelos, valor=1 si acierto, 0 si fallo
heatmap_data = []
for i in range(len(empresas)):
    fila = []
    for nombre_modelo in modelos.keys():
        fila.append(1 if resultados_por_modelo[nombre_modelo]["aciertos"][i] else 0)
    heatmap_data.append(fila)

heatmap_df = pd.DataFrame(heatmap_data, columns=list(modelos.keys()),
                           index=[f"{e['id']}. {e['nombre']}" for e in empresas])

# Anotaciones con prediccion y confianza
annot_data = []
for i in range(len(empresas)):
    fila = []
    for nombre_modelo in modelos.keys():
        pred = resultados_por_modelo[nombre_modelo]["predicciones"][i]
        conf = resultados_por_modelo[nombre_modelo]["confianzas"][i]
        fila.append(f"{pred}\n{conf:.0%}")
    annot_data.append(fila)

annot_df = pd.DataFrame(annot_data, columns=list(modelos.keys()),
                         index=heatmap_df.index)

cmap = sns.color_palette(["#fadbd8", "#d5f5e3"], as_cmap=True)
sns.heatmap(heatmap_df, annot=annot_df, fmt="", cmap=cmap, linewidths=1,
            linecolor="white", ax=ax, cbar=False, annot_kws={"fontsize": 9})

# Agregar columna de esperado como etiquetas
for i, esp in enumerate(esperados):
    ax.text(len(modelos) + 0.3, i + 0.5, esp, va="center", fontsize=10, fontweight="bold",
            color={"Alto": "#27ae60", "Medio": "#f39c12", "Bajo": "#e74c3c"}[esp])
ax.text(len(modelos) + 0.3, -0.3, "Esperado", fontsize=10, fontweight="bold", color="black")

ax.set_title("Predicciones por Empresa y Modelo\n(Verde=Acierto, Rojo=Fallo | Anotacion: Prediccion + Confianza)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("")
ax.set_xlabel("")
save_fig(fig, "28_heatmap_predicciones_empresa.png")

# --- 8.5 Barplot de accuracy por clase ---
print("[5/7] Accuracy por clase y modelo...")
fig, ax = plt.subplots(figsize=(12, 6))

clases_presentes = sorted(set(esperados))
x = np.arange(len(clases_presentes))
width = 0.25

for j, nombre_modelo in enumerate(modelos.keys()):
    accs_clase = []
    for clase in clases_presentes:
        idx = [i for i, e in enumerate(esperados) if e == clase]
        if idx:
            acc = sum(resultados_por_modelo[nombre_modelo]["aciertos"][i] for i in idx) / len(idx) * 100
        else:
            acc = 0
        accs_clase.append(acc)

    bars = ax.bar(x + j * width, accs_clase, width, label=nombre_modelo,
                  color=colores_modelo[nombre_modelo], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, accs_clase):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontweight="bold", fontsize=9)

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy por Clase en Datos de Produccion", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(clases_presentes, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 120)
ax.grid(axis="y", linestyle="--", alpha=0.4)
save_fig(fig, "29_accuracy_por_clase_produccion.png")

# --- 8.6 Radar resumen ejecutivo (Eficiencia vs Confiabilidad) ---
print("[6/7] Radar resumen ejecutivo...")

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

exec_metrics = ["Accuracy\nGeneral", "Confianza\nMedia", "Confianza\nMinima",
                "Consistencia\n(Aciertos/Total)", "Acc. Clase\nMayoritaria"]

for nombre_modelo in modelos.keys():
    info = [i for i in informe_modelos if i["Modelo"] == nombre_modelo][0]
    confs = resultados_por_modelo[nombre_modelo]["confianzas"]

    # Clase mayoritaria = Bajo (la mas representada)
    idx_bajo = [i for i, e in enumerate(esperados) if e == "Bajo"]
    acc_bajo = sum(resultados_por_modelo[nombre_modelo]["aciertos"][i] for i in idx_bajo) / len(idx_bajo) * 100

    vals = [
        info["Accuracy (%)"],
        info["Confianza Media (%)"],
        info["Confianza Min (%)"],
        info["Aciertos"] / info["Empresas"] * 100,
        acc_bajo,
    ]
    vals += vals[:1]

    exec_angles = np.linspace(0, 2 * np.pi, len(exec_metrics), endpoint=False).tolist()
    exec_angles += exec_angles[:1]

    ax.plot(exec_angles, vals, "o-", linewidth=2.5, markersize=8,
            label=nombre_modelo, color=colores_modelo[nombre_modelo])
    ax.fill(exec_angles, vals, alpha=0.1, color=colores_modelo[nombre_modelo])

ax.set_xticks(exec_angles[:-1])
ax.set_xticklabels(exec_metrics, fontsize=10, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="gray")
ax.set_title("Resumen Ejecutivo: Eficiencia vs Confiabilidad",
             fontsize=14, fontweight="bold", pad=25)
ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.02), fontsize=11,
          frameon=True, fancybox=True, shadow=True)
save_fig(fig, "30_radar_resumen_ejecutivo.png")

# --- 8.7 Tabla visual del informe completo ---
print("[7/7] Tabla visual del informe...")
fig, ax = plt.subplots(figsize=(18, 10))
ax.axis("off")

# Construir datos de tabla por empresa
header = ["ID", "Empresa", "Esperado",
          "Arbol\nPrediccion", "Arbol\nConfianza",
          "SVM\nPrediccion", "SVM\nConfianza",
          "RF\nPrediccion", "RF\nConfianza",
          "Consenso"]

cell_data = []
for i in range(len(empresas)):
    row = [
        str(ids_empresas[i]),
        nombres_empresas[i][:25],
        esperados[i],
    ]
    coincidencias = 0
    for nombre_modelo in modelos.keys():
        pred = resultados_por_modelo[nombre_modelo]["predicciones"][i]
        conf = resultados_por_modelo[nombre_modelo]["confianzas"][i]
        row.append(pred)
        row.append(f"{conf:.1%}")
        if resultados_por_modelo[nombre_modelo]["aciertos"][i]:
            coincidencias += 1
    row.append(f"{coincidencias}/{len(modelos)}")
    cell_data.append(row)

table = ax.table(cellText=cell_data, colLabels=header, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Estilo encabezado
for j in range(len(header)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colorear celdas de prediccion segun acierto/fallo
pred_col_indices = [3, 5, 7]  # Columnas de prediccion
modelo_names = list(modelos.keys())

for i in range(len(cell_data)):
    for k, col_idx in enumerate(pred_col_indices):
        nombre_m = modelo_names[k]
        if resultados_por_modelo[nombre_m]["aciertos"][i]:
            table[i + 1, col_idx].set_facecolor("#d5f5e3")
        else:
            table[i + 1, col_idx].set_facecolor("#fadbd8")

    # Colorear consenso
    consenso = cell_data[i][-1]
    n_ok = int(consenso.split("/")[0])
    if n_ok == 3:
        table[i + 1, len(header) - 1].set_facecolor("#27ae60")
        table[i + 1, len(header) - 1].set_text_props(color="white", fontweight="bold")
    elif n_ok >= 2:
        table[i + 1, len(header) - 1].set_facecolor("#f1c40f")
    else:
        table[i + 1, len(header) - 1].set_facecolor("#e74c3c")
        table[i + 1, len(header) - 1].set_text_props(color="white", fontweight="bold")

ax.set_title("Informe de Produccion: Predicciones, Confianza y Consenso por Empresa",
             fontsize=14, fontweight="bold", pad=25)
save_fig(fig, "31_informe_produccion_completo.png")

# =========================================================
# 9. RESUMEN FINAL
# =========================================================
print("\n" + "=" * 60)
print("  RESUMEN FINAL - PRUEBA EN PRODUCCION")
print("=" * 60)

print(f"\n  {'Modelo':<22} {'Accuracy':>10} {'Confianza':>12} {'Aciertos':>10}")
print(f"  {'-' * 54}")
mejor_acc = 0
mejor_nombre = ""
for info in informe_modelos:
    acc = info["Accuracy (%)"]
    if acc > mejor_acc:
        mejor_acc = acc
        mejor_nombre = info["Modelo"]
    marca = " <--" if acc == max(i["Accuracy (%)"] for i in informe_modelos) else ""
    print(f"  {info['Modelo']:<22} {info['Accuracy (%)']:>9.1f}% "
          f"{info['Confianza Media (%)']:>10.1f}%  "
          f"{info['Aciertos']}/{info['Empresas']}{marca}")

# Consenso
consenso_total = sum(
    1 for i in range(len(empresas))
    if all(resultados_por_modelo[m]["aciertos"][i] for m in modelos)
)
print(f"\n  Empresas con consenso unanime (3/3): {consenso_total}/{len(empresas)}")
print(f"  Modelo mas confiable: {mejor_nombre} ({mejor_acc:.1f}%)")

print(f"\n  Imagenes generadas en: {RESULTS_DIR}")
print("=" * 60)
print("\n  Archivos generados:")
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith(".png") and f[:2].isdigit() and int(f.split("_")[0]) >= 25:
        print(f"    - {f}")
print("\nPrueba en produccion completada exitosamente.")
