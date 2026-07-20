# =============================================================================
# entrenar.py — Entrenamiento dual del modelo predictivo de trayectorias
#
# Ejecutar una vez para generar ambos modelos entrenados:
#   py -3.10 entrenar.py
#
# Genera 2 modelos:
#   Modelo "Automata" (con OHE estado+transicion):
#     modelo/modelo_xgb.pkl, modelo/metadata.pkl
#     outputs/metrics.json, outputs/feature_importance.png, etc.
#
#   Modelo "Numerico" (sin leakage, solo datos academicos):
#     modelo/modelo_xgb_numerico.pkl, modelo/metadata_numerico.pkl
#     outputs/numerico/metrics.json, outputs/numerico/feature_importance.png, etc.
# =============================================================================

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES
# =============================================================================

RUTA_DATOS = 'datos/08_solo_pregrado_automata_corregido_validado_v2.xlsx'
DIR_MODELO = 'modelo'

UMBRAL_CURSOS = 55
UMBRAL_CREDITOS = 150

PROM_CREDITOS_GRADO = {
    'ADMINISTRACION DE EMPRESAS': 13.06,
    'CIENCIA POLITICA Y RELAC INTER': 13.75,
    'COMUNICACION SOCIAL': 18.67,
    'CONTADURIA PUBLICA': 15.83,
    'DERECHO': 13.5,
    'ECONOMIA': 19,
    'FINANZAS Y NEGOCIOS INTERNACIO': 15,
    'INGENIERIA AMBIENTAL': 14.6,
    'INGENIERIA BIOMEDICA': 14,
    'INGENIERIA CIVIL': 15,
    'INGENIERIA DE SISTEMAS': 13.5,
    'INGENIERIA DE SISTEMAS Y COMPU': 16.67,
    'INGENIERIA ELECTRICA': 38.5,
    'INGENIERIA ELECTRONICA': 15.4,
    'INGENIERIA INDUSTRIAL': 14.3,
    'INGENIERIA MECANICA': 14.67,
    'INGENIERIA MECATRONICA': 16.86,
    'INGENIERIA NAVAL': 15.82,
    'INGENIERIA QUIMICA': 16.67,
    'PSICOLOGIA': 18.26,
}

PROM_MATERIAS_GRADO = {
    'ADMINISTRACION DE EMPRESAS': 55.39,
    'CIENCIA POLITICA Y RELAC INTER': 50.5,
    'COMUNICACION SOCIAL': 60.74,
    'CONTADURIA PUBLICA': 56.42,
    'DERECHO': 64,
    'ECONOMIA': 54,
    'FINANZAS Y NEGOCIOS INTERNACIO': 58.25,
    'INGENIERIA AMBIENTAL': 56.2,
    'INGENIERIA BIOMEDICA': 58,
    'INGENIERIA CIVIL': 53.74,
    'INGENIERIA DE SISTEMAS': 55,
    'INGENIERIA DE SISTEMAS Y COMPU': 54.67,
    'INGENIERIA ELECTRICA': 43.07,
    'INGENIERIA ELECTRONICA': 55.32,
    'INGENIERIA INDUSTRIAL': 52.04,
    'INGENIERIA MECANICA': 53.06,
    'INGENIERIA MECATRONICA': 57.57,
    'INGENIERIA NAVAL': 57.45,
    'INGENIERIA QUIMICA': 59.33,
    'PSICOLOGIA': 46.64,
}

PROM_CREDITOS_DEFAULT = np.median(list(PROM_CREDITOS_GRADO.values()))
PROM_MATERIAS_DEFAULT = np.median(list(PROM_MATERIAS_GRADO.values()))

FEATURES_NUMERICAS = [
    'PROMEDIO_ACUMULADO', 'NRO_CURSOS_APROBADOS', 'CREDITOS_APROVADOS',
    'CURSOS_ACUM', 'CREDITOS_ACUM', 'ORDEN_AUTOMATA',
    'PORCENTAJE_CREDITOS_GRADO', 'PORCENTAJE_MATERIAS_GRADO',
]


def cargar_datos():
    print("Cargando dataset...")
    df = pd.read_excel(RUTA_DATOS)
    print(f"  Filas: {len(df):,}  |  Estudiantes: {df['ID'].nunique():,}")
    return df


def preprocesar(df):
    print("Preprocesando...")
    df = df.sort_values(['ID', 'PERIODO']).reset_index(drop=True)

    df['CURSOS_ACUM'] = (
        df.groupby('ID')['NRO_CURSOS_APROBADOS']
          .apply(lambda s: s.fillna(0).cumsum())
          .reset_index(drop=True)
    )
    df['CREDITOS_ACUM'] = (
        df.groupby('ID')['CREDITOS_APROVADOS']
          .apply(lambda s: s.fillna(0).cumsum())
          .reset_index(drop=True)
    )

    df['ORDEN_AUTOMATA'] = df.groupby('ID').cumcount() + 1

    prom_cred = df['PROGRAMA'].map(PROM_CREDITOS_GRADO).fillna(PROM_CREDITOS_DEFAULT)
    prom_mat = df['PROGRAMA'].map(PROM_MATERIAS_GRADO).fillna(PROM_MATERIAS_DEFAULT)
    df['PORCENTAJE_CREDITOS_GRADO'] = (df['CREDITOS_ACUM'] / prom_cred).clip(0, 1.5)
    df['PORCENTAJE_MATERIAS_GRADO'] = (df['CURSOS_ACUM'] / prom_mat).clip(0, 1.5)

    df['CUMPLE_REGLA_GRADO'] = (
        (df['CURSOS_ACUM'] > UMBRAL_CURSOS) &
        (df['CREDITOS_ACUM'] > UMBRAL_CREDITOS)
    ).astype(int)

    n_override = ((df['CUMPLE_REGLA_GRADO'] == 1) & (df['ESTADO_CONSECUENTE'] != 'Grado')).sum()
    n_rescatadas = ((df['CUMPLE_REGLA_GRADO'] == 1) & (df['ESTADO_CONSECUENTE'].isna())).sum()
    print(f"  Regla grado: {n_override:,} overrides, {n_rescatadas:,} rescatadas")

    df.loc[df['CUMPLE_REGLA_GRADO'] == 1, 'ESTADO_CONSECUENTE'] = 'Grado'

    df_clean = df.dropna(subset=['ESTADO_CONSECUENTE']).copy()
    print(f"  Dataset limpio: {len(df_clean):,} filas")
    return df_clean


def construir_features(df_clean, con_automata=True):
    nombre = "Automata (con OHE)" if con_automata else "Numerico (sin leakage)"
    print(f"\nConstruyendo features — {nombre}...")

    for col in FEATURES_NUMERICAS:
        df_clean[col] = df_clean.groupby('ESTADO_AUTOMATA')[col].transform(
            lambda x: x.fillna(x.median())
        )
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    parts = [df_clean[FEATURES_NUMERICAS].reset_index(drop=True)]

    if con_automata:
        FEATURES_BOOLEANAS = ['RIESGO_ACADEMICO', 'CUMPLE_REGLA_GRADO']
        df_clean['RIESGO_ACADEMICO'] = df_clean['ESTADO_AUTOMATA'].isin(
            ['PAP', 'PAT', 'PFU']
        ).astype(int)
        parts.append(df_clean[FEATURES_BOOLEANAS].reset_index(drop=True))

        estado_dummies = pd.get_dummies(df_clean['ESTADO_AUTOMATA'], prefix='ESTADO')
        trans_dummies = pd.get_dummies(df_clean['TRANSICION_AUTOMATA'], prefix='TRANS')
        parts.append(estado_dummies.reset_index(drop=True))
        parts.append(trans_dummies.reset_index(drop=True))

        desc = (f"{len(FEATURES_NUMERICAS)} num + {len(FEATURES_BOOLEANAS)} bool + "
                f"{estado_dummies.shape[1]} OHE estado + {trans_dummies.shape[1]} OHE trans")
    else:
        parts.append(df_clean[['CUMPLE_REGLA_GRADO']].reset_index(drop=True))
        desc = f"{len(FEATURES_NUMERICAS)} num + 1 bool (CUMPLE_REGLA_GRADO)"

    X = pd.concat(parts, axis=1)
    y = df_clean['ESTADO_CONSECUENTE'].reset_index(drop=True)
    groups = df_clean['ID'].reset_index(drop=True)

    print(f"  Features: {X.shape[1]} ({desc})")
    return X, y, groups


def cross_validate(X, y, groups, nombre, n_splits=5):
    print(f"\n{'='*70}")
    print(f"{n_splits}-FOLD CV — {nombre} — StratifiedGroupKFold")
    print(f"{'='*70}")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        ids_tr = set(groups.iloc[train_idx])
        ids_te = set(groups.iloc[test_idx])
        assert len(ids_tr & ids_te) == 0, f"Fuga de datos en fold {fold_i}"

        clases = sorted(y_tr.unique())
        c2i = {c: i for i, c in enumerate(clases)}
        i2c = {i: c for c, i in c2i.items()}
        y_tr_n = y_tr.map(c2i)
        y_te_n = y_te.map(c2i)

        min_c = y_tr_n.value_counts().min()
        kn = max(1, min(5, min_c - 1))

        X_res, y_res = SMOTE(random_state=42, k_neighbors=kn).fit_resample(X_tr, y_tr_n)

        m = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=42, objective='multi:softprob',
            eval_metric='mlogloss', verbosity=0,
        )
        m.fit(X_res, y_res, eval_set=[(X_te, y_te_n)], verbose=False)

        pred = pd.Series(m.predict(X_te)).map(i2c)
        acc = accuracy_score(y_te, pred)
        f1_m = f1_score(y_te, pred, average='macro', zero_division=0)
        f1_w = f1_score(y_te, pred, average='weighted', zero_division=0)

        fold_metrics.append({
            'fold': fold_i + 1, 'accuracy': acc,
            'f1_macro': f1_m, 'f1_weighted': f1_w,
        })

        print(f"  Fold {fold_i+1}: Acc={acc:.4f}  F1-Macro={f1_m:.4f}  F1-Weight={f1_w:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    cv_summary = {
        'accuracy_mean': metrics_df['accuracy'].mean(),
        'accuracy_std': metrics_df['accuracy'].std(),
        'f1_macro_mean': metrics_df['f1_macro'].mean(),
        'f1_macro_std': metrics_df['f1_macro'].std(),
        'f1_weighted_mean': metrics_df['f1_weighted'].mean(),
        'f1_weighted_std': metrics_df['f1_weighted'].std(),
    }

    print(f"\n  Consolidado:")
    print(f"    Accuracy:  {cv_summary['accuracy_mean']:.4f} +/- {cv_summary['accuracy_std']:.4f}")
    print(f"    F1 Macro:  {cv_summary['f1_macro_mean']:.4f} +/- {cv_summary['f1_macro_std']:.4f}")
    print(f"    F1 Weight: {cv_summary['f1_weighted_mean']:.4f} +/- {cv_summary['f1_weighted_std']:.4f}")

    return fold_metrics, cv_summary


def entrenar_modelo_final(X, y):
    print("\nEntrenando modelo final (todos los datos)...")
    clases = sorted(y.unique())
    c2i = {c: i for i, c in enumerate(clases)}
    y_n = y.map(c2i)

    min_c = y_n.value_counts().min()
    kn = max(1, min(5, min_c - 1))
    X_res, y_res = SMOTE(random_state=42, k_neighbors=kn).fit_resample(X, y_n)

    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, objective='multi:softprob',
        eval_metric='mlogloss', verbosity=0,
    )
    model.fit(X_res, y_res, verbose=False)
    print(f"  Modelo entrenado en {len(X_res):,} filas (tras SMOTE)")
    return model, c2i


def guardar_modelo(model, feature_names, c2i, cv_summary, fold_metrics,
                   n_rows, n_students, version_str, dir_modelo, dir_outputs):
    os.makedirs(dir_modelo, exist_ok=True)
    os.makedirs(dir_outputs, exist_ok=True)

    modelo_path = os.path.join(dir_modelo, 'modelo_xgb.pkl')
    with open(modelo_path, 'wb') as f:
        pickle.dump(model, f)

    metadata = {
        'feature_names': feature_names,
        'class_to_idx': c2i,
        'idx_to_class': {v: k for k, v in c2i.items()},
        'cv_summary': cv_summary,
        'fold_metrics': fold_metrics,
        'n_rows': n_rows,
        'n_students': n_students,
        'dataset': RUTA_DATOS,
    }
    meta_path = os.path.join(dir_modelo, 'metadata.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

    metrics_json = {
        'version': version_str,
        'dataset': RUTA_DATOS,
        'n_filas': n_rows,
        'n_estudiantes': n_students,
        'n_features': len(feature_names),
        'cv_results': {
            'n_splits': 5,
            'strategy': 'StratifiedGroupKFold (agrupado por ID)',
            'accuracy': {'mean': round(cv_summary['accuracy_mean'], 4),
                         'std': round(cv_summary['accuracy_std'], 4)},
            'f1_macro': {'mean': round(cv_summary['f1_macro_mean'], 4),
                         'std': round(cv_summary['f1_macro_std'], 4)},
            'f1_weighted': {'mean': round(cv_summary['f1_weighted_mean'], 4),
                            'std': round(cv_summary['f1_weighted_std'], 4)},
            'per_fold': fold_metrics,
        },
    }
    with open(os.path.join(dir_outputs, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Modelo guardado en {modelo_path}")
    print(f"[OK] Metadata guardada en {meta_path}")
    print(f"[OK] Metricas guardadas en {dir_outputs}/metrics.json")


def generar_figuras(model, feature_names, cv_summary, dir_outputs, titulo_extra=''):
    feat_imp = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top = feat_imp.head(15)
    ax.barh(top.index[::-1], top.values[::-1], color='#185FA5', edgecolor='white')
    ax.set_xlabel('Importancia (gain)', fontsize=11)
    ax.set_title(
        f'Importancia de Features{titulo_extra}\n'
        f'5-Fold CV Macro-F1 = {cv_summary["f1_macro_mean"]:.4f} '
        f'+/- {cv_summary["f1_macro_std"]:.4f}',
        fontsize=12, fontweight='bold',
    )
    ax.tick_params(axis='y', labelsize=9)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_outputs, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Figura guardada: {dir_outputs}/feature_importance.png")


def generar_matriz_confusion(model, X, y, groups, cv_summary, dir_outputs, titulo_extra=''):
    print("Generando matriz de confusion del ultimo fold...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(sgkf.split(X, y, groups=groups))
    train_idx, test_idx = folds[-1]

    X_te = X.iloc[test_idx]
    y_te = y.iloc[test_idx]

    pred_enc = model.predict(X_te)
    c2i = {c: i for i, c in enumerate(sorted(y.unique()))}
    i2c = {v: k for k, v in c2i.items()}
    y_pred = pd.Series(pred_enc).map(i2c)
    y_true = y_te.reset_index(drop=True)

    all_cls = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=all_cls)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annot_labels = np.array([
        f'{cm_norm[i, j]:.2f}\n({cm[i, j]:,})'
        for i in range(len(all_cls))
        for j in range(len(all_cls))
    ]).reshape(len(all_cls), len(all_cls))

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm_norm, annot=annot_labels, fmt='', cmap='Blues',
        xticklabels=all_cls, yticklabels=all_cls,
        square=True, linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.8}, annot_kws={'size': 8},
    )
    ax.set_title(
        f'Matriz de Confusion{titulo_extra} - Fold 5 de 5\n'
        f'XGBoost + SMOTE + Split por Estudiante\n'
        f'5-Fold CV Macro-F1 = {cv_summary["f1_macro_mean"]:.4f} '
        f'+/- {cv_summary["f1_macro_std"]:.4f}',
        fontsize=12, fontweight='bold', pad=15,
    )
    ax.set_xlabel('Estado Predicho', fontsize=11, labelpad=8)
    ax.set_ylabel('Estado Real', fontsize=11, labelpad=8)
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(dir_outputs, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    report_path = os.path.join(dir_outputs, 'classification_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Figura guardada: {dir_outputs}/confusion_matrix.png")
    print(f"[OK] Reporte guardado: {dir_outputs}/classification_report.json")


def entrenar_modelo(nombre, df_clean, con_automata, version_str, dir_modelo, dir_outputs):
    X, y, groups = construir_features(df_clean.copy(), con_automata=con_automata)
    fold_metrics, cv_summary = cross_validate(X, y, groups, nombre)
    model, c2i = entrenar_modelo_final(X, y)

    feature_names = list(X.columns)
    guardar_modelo(model, feature_names, c2i, cv_summary, fold_metrics,
                   len(df_clean), df_clean['ID'].nunique(),
                   version_str, dir_modelo, dir_outputs)
    generar_figuras(model, feature_names, cv_summary, dir_outputs, titulo_extra='')
    generar_matriz_confusion(model, X, y, groups, cv_summary, dir_outputs, titulo_extra='')

    return cv_summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ENTRENAMIENTO DUAL — Modelo Predictivo de Trayectorias Academicas")
    print("=" * 70)

    df = cargar_datos()
    df_clean = preprocesar(df)

    # --- Modelo 1: Automata (con OHE estado+transicion) ---
    print("\n" + "#" * 70)
    print("# MODELO 1: AUTOMATA (con OHE estado + transicion)")
    print("#" * 70)
    cv_auto = entrenar_modelo(
        nombre="Automata (con OHE)",
        df_clean=df_clean,
        con_automata=True,
        version_str="XGBoost + OHE(estado+transicion) + 5-fold CV (AUTOMATA)",
        dir_modelo='modelo',
        dir_outputs='outputs',
    )

    # --- Modelo 2: Numerico (sin leakage) ---
    print("\n" + "#" * 70)
    print("# MODELO 2: NUMERICO (sin leakage, solo datos academicos)")
    print("#" * 70)
    cv_num = entrenar_modelo(
        nombre="Numerico (sin leakage)",
        df_clean=df_clean,
        con_automata=False,
        version_str="XGBoost numerico + 5-fold CV (SIN LEAKAGE)",
        dir_modelo='modelo',
        dir_outputs='outputs/numerico',
    )

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO DUAL COMPLETADO")
    print("=" * 70)
    print(f"\n  Modelo Automata:  F1-Macro = {cv_auto['f1_macro_mean']:.4f} +/- {cv_auto['f1_macro_std']:.4f}")
    print(f"  Modelo Numerico:  F1-Macro = {cv_num['f1_macro_mean']:.4f} +/- {cv_num['f1_macro_std']:.4f}")
    print(f"\n  Gap: {cv_auto['f1_macro_mean'] - cv_num['f1_macro_mean']:.4f}")
    print(f"  (El gap representa la informacion que carries el estado/transicion del automata)")
    print(f"\n  Para ejecutar la app: py -3.10 -m streamlit run app.py")
