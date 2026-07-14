"""
==============================================================================
MODELO PREDICTIVO DE TRAYECTORIAS ACADÉMICAS A PARTIR DE AUTÓMATAS FINITOS
==============================================================================
Autores: [Andrés Aurelio Palmerin Torres - Diana Laura Ayala Maya]
Fecha  : 2025

Pipeline completo:
  Fase 3 — Construcción del conjunto de entrenamiento
  Fase 4 — Modelo predictivo + evaluación
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAPAS DEL AUTÓMATA
# ─────────────────────────────────────────────────────────────────────────────

# Transición → estado siguiente (según reglas del autómata)
TRANSICION_A_ESTADO = {
    'a': 'Continuo regular',
    'b': 'PAP/PAT',          # b desde CR→PAP, desde PAP→PAT
    'c': 'Grado',
    'k': 'PFU',
    'e': 'Recuperación académica',
    'g': 'Reingreso (solicitud)',
    'h': 'Reingreso',
    'i': 'Reinicio',
    'j': 'Primera vez en una carrera',
    'n': 'Primera vez en una carrera',
    'r': 'Transferencia interna',
    'f': 'Transferencia interna',
    's': 'Aspirante inscrito',
}

# Agrupación de estados para clases finales del clasificador
# (b desde PAP = PAT, b desde CR = PAP — usamos el estado SIGUIENTE real)
ESTADOS_VALIDOS_TARGET = [
    'Continuo regular',
    'PAP',
    'PAT',
    'Recuperación académica',
    'Grado',
    'PFU',
    'Reingreso',
    'Reinicio',
    'Primera vez en una carrera',
    'Transferencia interna',
]

print("=" * 70)
print("PIPELINE — MODELO PREDICTIVO DE TRAYECTORIAS ACADÉMICAS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/8] Cargando datos...")

df_main = pd.read_excel(
    'datos/12_only_undergraduate_with_automaton.xlsx',
    engine='openpyxl'
)
df_grad = pd.read_excel(
    'datos/07_undergraduate_pathway with degree automaton.xlsx',
    engine='openpyxl'
)

print(f"  Dataset principal : {df_main.shape[0]:,} filas × {df_main.shape[1]} columnas")
print(f"  Dataset graduados : {df_grad.shape[0]:,} filas × {df_grad.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO Y CONSTRUCCIÓN DEL CONJUNTO DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] Construyendo conjunto de entrenamiento...")

def build_training_set(df, source_label='main'):
    """
    Extrae filas de ACADEMIC_TRANSITION (una por periodo real),
    ordena por estudiante y ORDEN_AUTOMATA, y genera la columna
    ESTADO_SIGUIENTE mediante shift(-1) dentro de cada estudiante.
    """
    acad = df[df['REGLA_AUTOMATA'] == 'ACADEMIC_TRANSITION'].copy()
    acad = acad.sort_values(['ID', 'ORDEN_AUTOMATA']).reset_index(drop=True)

    # Estado siguiente = siguiente fila del mismo estudiante
    acad['ESTADO_SIGUIENTE'] = (
        acad.groupby('ID')['AUTOMATA_ESTADO']
            .shift(-1)
    )

    # Para filas donde el siguiente estado es NaN (último periodo del estudiante),
    # usamos la transición para inferir el estado siguiente
    mask_no_next = acad['ESTADO_SIGUIENTE'].isna()

    # Mapear transición al estado siguiente real
    trans_map_b = {
        'PAP'                      : 'PAT',
        'Primera vez en una carrera': 'PAP',
        'Continuo regular'          : 'PAP',
        'Recuperación académica'    : 'Exclusión',
        'PAT'                       : 'PAP',  # fallback
    }

    def infer_next_state(row):
        t = row['TRANSICION_AUTOMATA']
        estado = row['AUTOMATA_ESTADO']
        if t == 'a': return 'Continuo regular'
        if t == 'c': return 'Grado'
        if t == 'k': return 'PFU'
        if t == 'e': return 'Recuperación académica'
        if t == 'g': return 'PFU'
        if t == 'h': return 'Reingreso'
        if t == 'i': return 'Reinicio'
        if t == 'b': return trans_map_b.get(estado, 'PAP')
        return None

    acad.loc[mask_no_next, 'ESTADO_SIGUIENTE'] = (
        acad[mask_no_next].apply(infer_next_state, axis=1)
    )

    acad['_source'] = source_label
    return acad


# Construir desde ambos datasets
train_main = build_training_set(df_main, 'main')

# Para df_grad: alinear columnas comunes
common_cols = [c for c in df_main.columns if c in df_grad.columns]
train_grad = build_training_set(df_grad[common_cols], 'grad')

# Unir
df_train = pd.concat([train_main, train_grad], ignore_index=True)

# Eliminar filas sin target
df_train = df_train[df_train['ESTADO_SIGUIENTE'].notna()].copy()

print(f"  Filas totales con par (estado_actual, estado_siguiente): {len(df_train):,}")
print(f"  Estudiantes únicos: {df_train['ID'].nunique():,}")
print(f"\n  Distribución del target (ESTADO_SIGUIENTE):")
dist = df_train['ESTADO_SIGUIENTE'].value_counts()
for k, v in dist.items():
    pct = v / len(df_train) * 100
    print(f"    {k:<35} {v:>6,}  ({pct:5.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. INGENIERÍA DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Ingeniería de features...")

# Extraer año y semestre del periodo
df_train['ANIO'] = df_train['PERIODO'].astype(str).str[:4].astype(int)
df_train['SEMESTRE'] = df_train['PERIODO'].astype(str).str[4:].astype(int)

# Semestres desde ingreso
df_train['PERIODO_INGRESO_NUM'] = df_train['PERIODO_INGRESO'].astype(str).replace('', np.nan)
df_train['ANIO_INGRESO'] = df_train['PERIODO_INGRESO_NUM'].str[:4].astype(float)
df_train['SEM_INGRESO'] = df_train['PERIODO_INGRESO_NUM'].str[4:].astype(float)
df_train['PERIODOS_TRANSCURRIDOS'] = (
    (df_train['ANIO'] - df_train['ANIO_INGRESO']) * 2
    + (df_train['SEMESTRE'] - df_train['SEM_INGRESO'])
)

# Codificación ordinal del estado actual
ESTADO_RIESGO = {
    'Aspirante inscrito'        : 0,
    'Primera vez en una carrera': 1,
    'Continuo regular'          : 2,
    'PAP'                       : 3,
    'PAT'                       : 4,
    'Recuperación académica'    : 5,
    'Exclusión'                 : 6,
    'PFU'                       : 7,
    'Reingreso'                 : 2,
    'Reinicio'                  : 1,
    'Transferencia interna'     : 1,
    'Transferencia externa'     : 1,
    'Grado'                     : -1,
    'Final'                     : 7,
}
df_train['NIVEL_RIESGO_ESTADO'] = df_train['AUTOMATA_ESTADO'].map(ESTADO_RIESGO).fillna(0)

# Feature: flag de riesgo acumulado
df_train['RIESGO_EXCLUSION'] = df_train['RIESGO_EXCLUSION'].fillna(False).astype(int)
df_train['TUVO_RIESGO_EXCLUSION'] = df_train['TUVO_RIESGO_EXCLUSION'].fillna(False).astype(int)

# Porcentaje de avance en créditos y materias
for col in ['PORCENTAJE_CREDITOS_GRADO', 'PORCENTAJE_MATERIAS_GRADO']:
    if col not in df_train.columns:
        df_train[col] = np.nan

# Label encoding de variables categóricas
le_estado = LabelEncoder()
le_programa = LabelEncoder()
le_target = LabelEncoder()

df_train['ESTADO_ACTUAL_ENC'] = le_estado.fit_transform(df_train['AUTOMATA_ESTADO'].astype(str))
df_train['PROGRAMA_ENC'] = le_programa.fit_transform(df_train['PROGRAMA'].fillna('DESCONOCIDO').astype(str))
df_train['TARGET'] = le_target.fit_transform(df_train['ESTADO_SIGUIENTE'].astype(str))

TARGET_NAMES = le_target.classes_.tolist()
print(f"  Clases del target: {TARGET_NAMES}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SELECCIÓN DE FEATURES Y DIVISIÓN TRAIN/TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/8] Preparando matrices X, y...")

FEATURE_COLS = [
    'ESTADO_ACTUAL_ENC',
    'NIVEL_RIESGO_ESTADO',
    'PROMEDIO',
    'PROMEDIO_ACUMULADO',
    'NRO_CURSOS_APROBADOS',
    'CREDITOS_APROVADOS',
    'PERIODOS_TRANSCURRIDOS',
    'PROGRAMA_ENC',
    'RIESGO_EXCLUSION',
    'TUVO_RIESGO_EXCLUSION',
    'PORCENTAJE_CREDITOS_GRADO',
    'PORCENTAJE_MATERIAS_GRADO',
]

# Solo columnas disponibles
FEATURE_COLS = [c for c in FEATURE_COLS if c in df_train.columns]

X = df_train[FEATURE_COLS].copy()
y = df_train['TARGET'].copy()

print(f"  Features seleccionados: {len(FEATURE_COLS)}")
print(f"  Filas usables: {len(X):,}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRENAMIENTO DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/8] Entrenando modelos...")

imputer = SimpleImputer(strategy='median')

models = {
    'Random Forest': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_leaf=2, class_weight='balanced',
            random_state=42, n_jobs=-1
        ))
    ]),
    'Gradient Boosting': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(
            n_estimators=150, max_depth=5,
            learning_rate=0.1, random_state=42
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=42, C=1.0
        ))
    ]),
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='mlogloss', random_state=42,
            n_jobs=-1
        ))
    ])

results = {}
for name, pipe in models.items():
    print(f"  Entrenando {name}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    results[name] = {
        'pipe': pipe,
        'y_pred': y_pred,
        'accuracy': acc,
        'f1_weighted': f1,
        'precision_weighted': prec,
        'recall_weighted': rec,
    }
    print(f"    Accuracy={acc:.4f}  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")

best_name = max(results, key=lambda n: results[n]['f1_weighted'])
best = results[best_name]
print(f"\n  ✓ Mejor modelo: {best_name}  (F1={best['f1_weighted']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUACIÓN DETALLADA DEL MEJOR MODELO
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6/8] Evaluación detallada — {best_name}...")

y_pred_best = best['y_pred']
report = classification_report(
    y_test, y_pred_best,
    target_names=TARGET_NAMES,
    zero_division=0,
    output_dict=True
)

print("\n  Reporte por clase:")
header = f"  {'Estado':<35} {'Precision':>9} {'Recall':>8} {'F1':>8} {'Soporte':>9}"
print(header)
print("  " + "-" * (len(header) - 2))
for cls in TARGET_NAMES:
    if cls in report:
        r = report[cls]
        print(f"  {cls:<35} {r['precision']:>9.3f} {r['recall']:>8.3f} {r['f1-score']:>8.3f} {int(r['support']):>9,}")

# Guardar métricas
metrics_out = {
    'best_model': best_name,
    'accuracy': float(best['accuracy']),
    'f1_weighted': float(best['f1_weighted']),
    'precision_weighted': float(best['precision_weighted']),
    'recall_weighted': float(best['recall_weighted']),
    'all_models': {
        n: {k: float(v) for k, v in r.items() if k != 'pipe' and k != 'y_pred'}
        for n, r in results.items()
    },
    'classification_report': report,
    'target_names': TARGET_NAMES,
    'feature_names': FEATURE_COLS,
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
}

with open(f'{OUTPUT_DIR}/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_out, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Generando visualizaciones...")

PALETTE = {
    'Continuo regular'           : '#1D9E75',
    'PAP'                        : '#EF9F27',
    'PAT'                        : '#D85A30',
    'Recuperación académica'     : '#BA7517',
    'Grado'                      : '#185FA5',
    'PFU'                        : '#888780',
    'Reingreso'                  : '#534AB7',
    'Reinicio'                   : '#D4537E',
    'Primera vez en una carrera' : '#5DCAA5',
    'Transferencia interna'      : '#997756',
    'Exclusión'                  : '#E24B4A',
    'Aspirante inscrito'         : '#C0DD97',
}

# ── Fig 1: Comparación de modelos ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Comparación de Modelos Predictivos', fontsize=14, fontweight='bold', y=1.02)

model_names = list(results.keys())
metrics_list = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
metric_labels = ['Accuracy', 'F1 (weighted)', 'Precision (weighted)', 'Recall (weighted)']
colors_models = ['#185FA5', '#1D9E75', '#EF9F27', '#D85A30'][:len(model_names)]

# Barras por accuracy
ax = axes[0]
vals = [results[n]['accuracy'] for n in model_names]
bars = ax.barh(model_names, vals, color=colors_models, height=0.5)
ax.set_xlim(0, 1.05)
ax.set_xlabel('Accuracy')
ax.set_title('Accuracy por modelo')
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=10)
ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Barras por F1
ax = axes[1]
vals_f1 = [results[n]['f1_weighted'] for n in model_names]
bars2 = ax.barh(model_names, vals_f1, color=colors_models, height=0.5)
ax.set_xlim(0, 1.05)
ax.set_xlabel('F1 Score (weighted)')
ax.set_title('F1 Score por modelo')
ax.bar_label(bars2, fmt='%.4f', padding=3, fontsize=10)
ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_comparacion_modelos.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_comparacion_modelos.png")

# ── Fig 2: Matriz de confusión ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_best)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle(f'Matriz de Confusión — {best_name}', fontsize=14, fontweight='bold')

for ax, data, title, fmt in zip(
    axes,
    [cm, cm_norm],
    ['Conteos absolutos', 'Proporción por clase real'],
    ['d', '.2f']
):
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
        ax=ax, linewidths=0.4, linecolor='white',
        cbar_kws={'shrink': 0.8}
    )
    ax.set_xlabel('Estado predicho', fontsize=11)
    ax.set_ylabel('Estado real', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_confusion_matrix.png")

# ── Fig 3: F1 por clase ────────────────────────────────────────────────────
f1_per_class = {
    cls: report[cls]['f1-score']
    for cls in TARGET_NAMES
    if cls in report
}
sorted_classes = sorted(f1_per_class, key=f1_per_class.get, reverse=True)
f1_vals = [f1_per_class[c] for c in sorted_classes]
bar_colors = [PALETTE.get(c, '#888780') for c in sorted_classes]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(sorted_classes, f1_vals, color=bar_colors, width=0.65, edgecolor='white', linewidth=0.5)
ax.set_ylim(0, 1.1)
ax.set_ylabel('F1 Score')
ax.set_title(f'F1 Score por Estado Académico — {best_name}', fontsize=13, fontweight='bold')
ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, label='Umbral 0.70')
ax.axhline(y=float(best['f1_weighted']), color='#185FA5', linestyle='--', alpha=0.6,
           linewidth=1.0, label=f'F1 global ({best["f1_weighted"]:.3f})')
ax.tick_params(axis='x', rotation=35, labelsize=9)
ax.legend(fontsize=9)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_f1_por_clase.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_f1_por_clase.png")

# ── Fig 4: Importancia de features ────────────────────────────────────────
best_pipe = best['pipe']
if hasattr(best_pipe.named_steps.get('clf', best_pipe[-1]), 'feature_importances_'):
    clf_step = best_pipe.named_steps.get('clf', best_pipe[-1])
    importances = clf_step.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_imp = ['#185FA5' if v >= feat_imp.quantile(0.75) else '#B5D4F4' for v in feat_imp]
    bars = ax.barh(feat_imp.index, feat_imp.values, color=colors_imp, height=0.6)
    ax.set_xlabel('Importancia relativa')
    ax.set_title(f'Importancia de Features — {best_name}', fontsize=13, fontweight='bold')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_feature_importance.png")

# ── Fig 5: Distribución del target ────────────────────────────────────────
target_counts = df_train['ESTADO_SIGUIENTE'].value_counts()
colors_dist = [PALETTE.get(c, '#888780') for c in target_counts.index]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(target_counts.index, target_counts.values, color=colors_dist,
              width=0.7, edgecolor='white', linewidth=0.5)
ax.set_ylabel('Cantidad de registros')
ax.set_title('Distribución del Estado Siguiente (Target) en el Dataset de Entrenamiento',
             fontsize=12, fontweight='bold')
ax.bar_label(bars, fmt='%d', padding=3, fontsize=9)
ax.tick_params(axis='x', rotation=35, labelsize=9)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_distribucion_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 05_distribucion_target.png")

# ── Fig 6: Trayectorias más frecuentes ────────────────────────────────────
print("\n  Calculando trayectorias más frecuentes...")

# Reconstruir trayectorias por estudiante
all_data = pd.concat([df_main, df_grad[[c for c in df_main.columns if c in df_grad.columns]]],
                      ignore_index=True)
acad_all = all_data[all_data['REGLA_AUTOMATA'] == 'ACADEMIC_TRANSITION'].copy()
acad_all = acad_all.sort_values(['ID', 'ORDEN_AUTOMATA'])

trayectorias = (
    acad_all.groupby('ID')['AUTOMATA_ESTADO']
    .apply(lambda x: ' → '.join(x.tolist()))
    .reset_index()
    .rename(columns={'AUTOMATA_ESTADO': 'TRAYECTORIA'})
)

top_trayectorias = trayectorias['TRAYECTORIA'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(13, 6))
y_pos = range(len(top_trayectorias))
colors_tray = ['#185FA5' if 'Grado' in t or 'Continuo' in t else
               '#D85A30' if 'PAT' in t or 'PFU' in t else
               '#1D9E75'
               for t in top_trayectorias.index]

bars = ax.barh(list(y_pos), top_trayectorias.values, color=colors_tray,
               height=0.65, edgecolor='white', linewidth=0.3)

# Truncar trayectorias largas para el label
truncated_labels = [
    t if len(t) <= 80 else t[:77] + '...'
    for t in top_trayectorias.index
]
ax.set_yticks(list(y_pos))
ax.set_yticklabels(truncated_labels, fontsize=7.5)
ax.set_xlabel('Número de estudiantes')
ax.set_title('Top 15 Trayectorias Académicas más Frecuentes', fontsize=13, fontweight='bold')
ax.bar_label(bars, fmt='%d', padding=3, fontsize=9)

legend_elements = [
    mpatches.Patch(color='#185FA5', label='Trayectoria hacia Grado / Regular'),
    mpatches.Patch(color='#D85A30', label='Trayectoria con riesgo (PAT / PFU)'),
    mpatches.Patch(color='#1D9E75', label='Otras trayectorias'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_trayectorias_frecuentes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 06_trayectorias_frecuentes.png")

# ── Fig 7: Diagrama visual del autómata ───────────────────────────────────
# (SVG generado por el notebook, aquí hacemos versión matplotlib)
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title('Autómata Finito de Trayectorias Académicas', fontsize=14, fontweight='bold', pad=15)

NODE_POSITIONS = {
    'Aspirante inscrito'        : (1,   8),
    'Primera vez en una carrera': (3.5, 8),
    'Continuo regular'          : (7,   6.5),
    'PAP'                       : (7,   4.5),
    'PAT'                       : (7,   2.5),
    'Recuperación académica'    : (10,  3.5),
    'Exclusión'                 : (10,  1.5),
    'Grado'                     : (12,  6.5),
    'PFU'                       : (4,   3.5),
    'Reingreso'                 : (4,   5.5),
    'Reinicio'                  : (1.5, 5.5),
    'Transferencia interna'     : (1.5, 7),
}

NODE_COLORS = {
    'Aspirante inscrito'        : '#C0DD97',
    'Primera vez en una carrera': '#5DCAA5',
    'Continuo regular'          : '#1D9E75',
    'PAP'                       : '#EF9F27',
    'PAT'                       : '#D85A30',
    'Recuperación académica'    : '#BA7517',
    'Exclusión'                 : '#E24B4A',
    'Grado'                     : '#185FA5',
    'PFU'                       : '#888780',
    'Reingreso'                 : '#534AB7',
    'Reinicio'                  : '#D4537E',
    'Transferencia interna'     : '#9FE1CB',
}

for state, (x, y) in NODE_POSITIONS.items():
    color = NODE_COLORS.get(state, '#cccccc')
    circle = plt.Circle((x, y), 0.6, color=color, zorder=3, alpha=0.9)
    ax.add_patch(circle)
    label = state.replace(' ', '\n') if len(state) > 14 else state
    fontsize = 6 if len(state) > 18 else 7
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=4, color='white' if color in
            ['#1D9E75','#185FA5','#534AB7','#D85A30','#E24B4A','#BA7517','#888780'] else '#333')

EDGES = [
    ('Aspirante inscrito', 'Primera vez en una carrera', 'n'),
    ('Primera vez en una carrera', 'Continuo regular', 'a'),
    ('Primera vez en una carrera', 'PAP', 'b'),
    ('Primera vez en una carrera', 'PFU', 'k'),
    ('Continuo regular', 'Continuo regular', 'a'),
    ('Continuo regular', 'PAP', 'b'),
    ('Continuo regular', 'Grado', 'c'),
    ('Continuo regular', 'PFU', 'k'),
    ('PAP', 'Continuo regular', 'a'),
    ('PAP', 'PAT', 'b'),
    ('PAT', 'Continuo regular', 'a'),
    ('PAT', 'Recuperación académica', 'e'),
    ('Recuperación académica', 'Continuo regular', 'a'),
    ('Recuperación académica', 'Exclusión', 'd'),
    ('PFU', 'Reingreso', 'g/h'),
    ('PFU', 'Reinicio', 'i'),
    ('Reingreso', 'Continuo regular', 'h'),
    ('Exclusión', 'Reinicio', 'i'),
    ('Reinicio', 'Primera vez en una carrera', 'j'),
]

arrowprops = dict(arrowstyle='->', color='#444', lw=0.8,
                  connectionstyle='arc3,rad=0.15')

for src, dst, label in EDGES:
    if src == dst:
        continue
    x1, y1 = NODE_POSITIONS.get(src, (0, 0))
    x2, y2 = NODE_POSITIONS.get(dst, (0, 0))
    if x1 == 0 or x2 == 0:
        continue
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=arrowprops, zorder=2)
    mx, my = (x1 + x2) / 2 + 0.1, (y1 + y2) / 2 + 0.15
    ax.text(mx, my, label, fontsize=6, color='#333', ha='center',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_automata_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 07_automata_diagram.png")

# ── Fig 8: Métricas por modelo (tabla visual) ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table_data = [
    [n,
     f"{results[n]['accuracy']:.4f}",
     f"{results[n]['precision_weighted']:.4f}",
     f"{results[n]['recall_weighted']:.4f}",
     f"{results[n]['f1_weighted']:.4f}"]
    for n in model_names
]
col_labels = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1 Weighted']
tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.4, 2.0)

# Highlight best row
best_idx = model_names.index(best_name) + 1
for col_idx in range(len(col_labels)):
    tbl[best_idx, col_idx].set_facecolor('#E6F1FB')
    tbl[best_idx, col_idx].set_text_props(fontweight='bold')
for col_idx in range(len(col_labels)):
    tbl[0, col_idx].set_facecolor('#185FA5')
    tbl[0, col_idx].set_text_props(color='white', fontweight='bold')

ax.set_title('Resumen de Métricas por Modelo', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_tabla_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 08_tabla_metricas.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORTAR DATASET DE ENTRENAMIENTO + RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] Exportando resultados...")

# Dataset de entrenamiento
export_cols = [
    'ID', 'PERIODO', 'PROGRAMA', 'AUTOMATA_ESTADO', 'TRANSICION_AUTOMATA',
    'PROMEDIO', 'PROMEDIO_ACUMULADO', 'NRO_CURSOS_APROBADOS', 'CREDITOS_APROVADOS',
    'PERIODOS_TRANSCURRIDOS', 'RIESGO_EXCLUSION', 'TUVO_RIESGO_EXCLUSION',
    'ESTADO_SIGUIENTE'
]
export_cols = [c for c in export_cols if c in df_train.columns]
df_train[export_cols].to_csv(
    f'{OUTPUT_DIR}/training_dataset.csv', index=False, encoding='utf-8-sig'
)
print(f"  ✓ training_dataset.csv  ({len(df_train):,} filas)")

# Trayectorias por estudiante
trayectorias_full = (
    acad_all.groupby('ID').agg(
        PROGRAMA=('PROGRAMA', 'first'),
        N_PERIODOS=('PERIODO', 'count'),
        SECUENCIA_ESTADOS=('AUTOMATA_ESTADO', lambda x: ' → '.join(x.tolist())),
        SECUENCIA_TRANSICIONES=('TRANSICION_AUTOMATA', lambda x: ''.join(x.fillna('?').tolist())),
        ESTADO_FINAL=('AUTOMATA_ESTADO', 'last'),
    )
).reset_index()

trayectorias_full.to_csv(
    f'{OUTPUT_DIR}/trayectorias_por_estudiante.csv', index=False, encoding='utf-8-sig'
)
print(f"  ✓ trayectorias_por_estudiante.csv  ({len(trayectorias_full):,} estudiantes)")

# Resumen preguntas de investigación
print("\n" + "=" * 70)
print("RESPUESTAS A LAS PREGUNTAS DE INVESTIGACIÓN")
print("=" * 70)

tray_count = trayectorias_full['SECUENCIA_ESTADOS'].value_counts()
print(f"\n1. Trayectoria más frecuente ({tray_count.iloc[0]} estudiantes):")
print(f"   {tray_count.index[0]}")

grad_mask = trayectorias_full['ESTADO_FINAL'] == 'Grado'
print(f"\n2. Estudiantes que terminan en Grado: {grad_mask.sum():,} "
      f"({grad_mask.mean()*100:.1f}%)")

riesgo_mask = trayectorias_full['ESTADO_FINAL'].isin(['PAP','PAT','PFU','Final'])
print(f"   Estudiantes en riesgo o PFU: {riesgo_mask.sum():,} "
      f"({riesgo_mask.mean()*100:.1f}%)")

if hasattr(best_pipe.named_steps.get('clf', best_pipe[-1]), 'feature_importances_'):
    clf_step_q = best_pipe.named_steps.get('clf', best_pipe[-1])
    top_feat = pd.Series(clf_step_q.feature_importances_, index=FEATURE_COLS).nlargest(3)
    print(f"\n3. Variables más importantes para la predicción:")
    for feat, imp in top_feat.items():
        print(f"   {feat:<40} {imp:.4f}")

print(f"\n4. Desempeño del modelo ({best_name}):")
print(f"   Accuracy  : {best['accuracy']:.4f}")
print(f"   F1 (w)    : {best['f1_weighted']:.4f}")
print(f"   Precision : {best['precision_weighted']:.4f}")
print(f"   Recall    : {best['recall_weighted']:.4f}")

print("\n" + "=" * 70)
print(f"Archivos generados en: {OUTPUT_DIR}/")
print("=" * 70)
