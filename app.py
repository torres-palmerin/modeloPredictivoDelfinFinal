# =============================================================================
# app.py — Simulador Interactivo del Modelo Predictivo de Trayectorias
#
# Ejecutar despues de entrenar el modelo:
#   py -3.10 -m streamlit run app.py
#
# Requiere que entrenar.py haya sido ejecutado primero para generar
# ambos modelos (automata + numerico)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import json
import pickle

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

st.set_page_config(
    page_title="Modelo Predictivo de Trayectorias",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ESTILOS CSS
# =============================================================================

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #185FA5 0%, #0D3D6E 100%);
    padding: 2rem 2.5rem; border-radius: 14px;
    margin-bottom: 1.8rem; color: white;
}
.main-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 0.4rem; }
.main-header p  { font-size: 0.95rem; opacity: 0.85; margin: 0; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    padding: 3px 12px; border-radius: 20px; font-size: 0.78rem;
    font-weight: 600; margin-top: 0.7rem; letter-spacing: 0.04em;
}
.metric-card {
    border-radius: 12px; padding: 1.3rem 1.5rem;
    text-align: center; border: 1px solid #E2E8F0;
}
.metric-card .metric-value { font-size: 2.4rem; font-weight: 800; line-height: 1; margin: 0.3rem 0; }
.metric-card .metric-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; opacity: 0.7; }
.metric-card .metric-sub { font-size: 0.82rem; opacity: 0.6; margin-top: 0.3rem; }
.result-card {
    border-radius: 14px; padding: 1.8rem 2rem;
    margin-top: 1.2rem; border-left: 6px solid;
}
.result-card .result-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; opacity: 0.7; margin-bottom: 0.4rem; }
.result-card .result-estado { font-size: 2rem; font-weight: 800; line-height: 1.1; margin: 0.2rem 0; }
.result-card .result-trans { font-size: 0.88rem; opacity: 0.75; margin-top: 0.5rem; font-family: monospace; }
div.stButton > button {
    background: #185FA5; color: white; font-weight: 600; font-size: 0.95rem;
    padding: 0.6rem 2rem; border-radius: 10px; border: none; width: 100%; transition: background 0.2s;
}
div.stButton > button:hover { background: #0D3D6E; }
.section-title {
    font-size: 1.05rem; font-weight: 700; color: #185FA5;
    border-bottom: 2px solid #185FA5; padding-bottom: 0.4rem;
    margin: 1.4rem 0 1rem; letter-spacing: 0.02em;
}
.prediction-result-box {
    background: #F8FAFC; border: 2px solid #185FA5; border-radius: 14px;
    padding: 1.8rem 2rem; margin-top: 1rem; text-align: center;
}
.prediction-result-box .pred-label {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 700; color: #64748B; margin-bottom: 0.3rem;
}
.prediction-result-box .pred-estado {
    font-size: 2.6rem; font-weight: 900; line-height: 1.1; margin: 0.2rem 0;
}
.prediction-result-box .pred-conf {
    font-size: 1.1rem; font-weight: 700; margin-top: 0.6rem;
}
.prediction-result-box .pred-trans {
    font-size: 0.9rem; color: #64748B; margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTES
# =============================================================================

PALETTE = {
    "Continuo regular": "#1D9E75", "PAP": "#EF9F27", "PAT": "#D85A30",
    "Recuperación académica": "#BA7517", "Grado": "#185FA5", "PFU": "#888780",
    "Reingreso": "#534AB7", "Reinicio": "#D4537E",
    "Primera vez en una carrera": "#5DCAA5", "Transferencia interna": "#997756",
    "Exclusión": "#E24B4A", "Final": "#A05050",
    "Aspirante inscrito": "#5DCAA5", "Transferencia externa": "#997756",
}

BG_PALETTE = {
    "Grado": "#EBF3FC", "Continuo regular": "#EDFAF4", "Final": "#F9F0F0",
    "PAP": "#FEF6E8", "PAT": "#FDEEE8", "Recuperación académica": "#FEF3E2",
    "PFU": "#F3F3F2", "Reingreso": "#EFEDFD", "Reinicio": "#FDEEF5",
    "Primera vez en una carrera": "#EDFAF7", "Transferencia interna": "#F8F4EE",
    "Aspirante inscrito": "#EDFAF7", "Exclusión": "#FEE8E8",
    "Transferencia externa": "#F8F4EE",
}

TRANS_DESCRIPCIONES = {
    "a": "Continúa regular", "b": "Riesgo académico", "c": "Graduación",
    "d": "Exclusión", "e": "Recuperación", "f": "Transferencia interna",
    "g": "Solicitud reingreso", "h": "Reingreso aprobado", "i": "Reinicio",
    "j": "Reinicio aprobado", "k": "PFU (retiro/ausencia)",
    "n": "Ingreso inicial", "r": "Transferencia interna", "s": "Transferencia externa",
}

UMBRAL_CURSOS = 55
UMBRAL_CREDITOS = 150

PROM_CREDITOS_GRADO = {
    'ADMINISTRACION DE EMPRESAS': 13.06, 'CIENCIA POLITICA Y RELAC INTER': 13.75,
    'COMUNICACION SOCIAL': 18.67, 'CONTADURIA PUBLICA': 15.83, 'DERECHO': 13.5,
    'ECONOMIA': 19, 'FINANZAS Y NEGOCIOS INTERNACIO': 15,
    'INGENIERIA AMBIENTAL': 14.6, 'INGENIERIA BIOMEDICA': 14,
    'INGENIERIA CIVIL': 15, 'INGENIERIA DE SISTEMAS': 13.5,
    'INGENIERIA DE SISTEMAS Y COMPU': 16.67, 'INGENIERIA ELECTRICA': 38.5,
    'INGENIERIA ELECTRONICA': 15.4, 'INGENIERIA INDUSTRIAL': 14.3,
    'INGENIERIA MECANICA': 14.67, 'INGENIERIA MECATRONICA': 16.86,
    'INGENIERIA NAVAL': 15.82, 'INGENIERIA QUIMICA': 16.67, 'PSICOLOGIA': 18.26,
}

PROM_MATERIAS_GRADO = {
    'ADMINISTRACION DE EMPRESAS': 55.39, 'CIENCIA POLITICA Y RELAC INTER': 50.5,
    'COMUNICACION SOCIAL': 60.74, 'CONTADURIA PUBLICA': 56.42, 'DERECHO': 64,
    'ECONOMIA': 54, 'FINANZAS Y NEGOCIOS INTERNACIO': 58.25,
    'INGENIERIA AMBIENTAL': 56.2, 'INGENIERIA BIOMEDICA': 58,
    'INGENIERIA CIVIL': 53.74, 'INGENIERIA DE SISTEMAS': 55,
    'INGENIERIA DE SISTEMAS Y COMPU': 54.67, 'INGENIERIA ELECTRICA': 43.07,
    'INGENIERIA ELECTRONICA': 55.32, 'INGENIERIA INDUSTRIAL': 52.04,
    'INGENIERIA MECANICA': 53.06, 'INGENIERIA MECATRONICA': 57.57,
    'INGENIERIA NAVAL': 57.45, 'INGENIERIA QUIMICA': 59.33, 'PSICOLOGIA': 46.64,
}

PROM_CREDITOS_DEFAULT = np.median(list(PROM_CREDITOS_GRADO.values()))
PROM_MATERIAS_DEFAULT = np.median(list(PROM_MATERIAS_GRADO.values()))


# =============================================================================
# CARGA DE MODELOS
# =============================================================================

@st.cache_resource
def load_models():
    paths = {
        'automata': ('modelo/modelo_xgb.pkl', 'modelo/metadata.pkl'),
        'numerico': ('modelo/modelo_xgb_numerico.pkl', 'modelo/metadata_numerico.pkl'),
    }
    models = {}
    for key, (mp, mtp) in paths.items():
        if not os.path.exists(mp) or not os.path.exists(mtp):
            st.error(f"Modelo '{key}' no encontrado. Ejecuta: `py -3.10 entrenar.py`")
            st.stop()
        with open(mp, 'rb') as f:
            model = pickle.load(f)
        with open(mtp, 'rb') as f:
            metadata = pickle.load(f)
        models[key] = (model, metadata)
    return models


@st.cache_resource
def load_estado_options():
    RUTA = 'datos/08_solo_pregrado_automata_corregido_validado_v2.xlsx'
    if not os.path.exists(RUTA):
        return (
            ['Continuo regular', 'PAP', 'PAT', 'PFU', 'Grado',
             'Reingreso', 'Reinicio', 'Transferencia interna',
             'Transferencia externa', 'Exclusión', 'Final',
             'Aspirante inscrito', 'Primera vez en una carrera',
             'Recuperación académica'],
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'n', 'r', 's'],
            [],
        )
    df = pd.read_excel(RUTA, usecols=['ESTADO_AUTOMATA', 'TRANSICION_AUTOMATA', 'PROGRAMA'])
    return (
        sorted(df['ESTADO_AUTOMATA'].dropna().unique().tolist()),
        sorted(df['TRANSICION_AUTOMATA'].dropna().unique().tolist()),
        sorted(df['PROGRAMA'].dropna().unique().tolist()),
    )


def predict_individual(model, metadata, estado, transicion,
                       ppa, cursos_aprob, creditos_aprob, orden, programa,
                       con_automata=True):
    feature_names = metadata['feature_names']
    idx_to_class = metadata['idx_to_class']
    cursos_acum = float(cursos_aprob)
    creditos_acum = float(creditos_aprob)
    prom_cred = PROM_CREDITOS_GRADO.get(programa, PROM_CREDITOS_DEFAULT)
    prom_mat = PROM_MATERIAS_GRADO.get(programa, PROM_MATERIAS_DEFAULT)

    input_dict = {
        'PROMEDIO_ACUMULADO': ppa,
        'NRO_CURSOS_APROBADOS': float(cursos_aprob),
        'CREDITOS_APROVADOS': float(creditos_aprob),
        'CURSOS_ACUM': cursos_acum,
        'CREDITOS_ACUM': creditos_acum,
        'ORDEN_AUTOMATA': float(orden),
        'PORCENTAJE_CREDITOS_GRADO': min(creditos_acum / prom_cred, 1.5),
        'PORCENTAJE_MATERIAS_GRADO': min(cursos_acum / prom_mat, 1.5),
        'CUMPLE_REGLA_GRADO': float(cursos_acum > UMBRAL_CURSOS and creditos_acum > UMBRAL_CREDITOS),
    }

    if con_automata:
        input_dict['RIESGO_ACADEMICO'] = float(estado in ['PAP', 'PAT', 'PFU'])

    row = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for col, val in input_dict.items():
        if col in row.columns:
            row[col] = val

    if con_automata:
        est_col = f'ESTADO_{estado}'
        if est_col in row.columns:
            row[est_col] = 1
        trans_col = f'TRANS_{transicion}'
        if trans_col in row.columns:
            row[trans_col] = 1

    pred_encoded = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    pred_name = idx_to_class[int(pred_encoded)]
    proba_dict = {idx_to_class[i]: float(p) for i, p in enumerate(proba)}
    return pred_name, proba_dict


def predict_batch(model, metadata, df, con_automata=True):
    feature_names = metadata['feature_names']
    X = pd.DataFrame(0.0, index=range(len(df)), columns=feature_names)

    for src in ['PROMEDIO_ACUMULADO', 'NRO_CURSOS_APROBADOS', 'CREDITOS_APROVADOS']:
        if src in df.columns and src in X.columns:
            X[src] = df[src].fillna(0).astype(float)

    if 'CURSOS_ACUM' in X.columns:
        X['CURSOS_ACUM'] = X['NRO_CURSOS_APROBADOS'].cumsum()
    if 'CREDITOS_ACUM' in X.columns:
        X['CREDITOS_ACUM'] = X['CREDITOS_APROVADOS'].cumsum()
    if 'ORDEN_AUTOMATA' in X.columns:
        X['ORDEN_AUTOMATA'] = np.arange(1, len(df) + 1, dtype=float)

    if 'PROGRAMA' in df.columns:
        prom_c = df['PROGRAMA'].map(PROM_CREDITOS_GRADO).fillna(PROM_CREDITOS_DEFAULT)
        prom_m = df['PROGRAMA'].map(PROM_MATERIAS_GRADO).fillna(PROM_MATERIAS_DEFAULT)
        if 'PORCENTAJE_CREDITOS_GRADO' in X.columns:
            X['PORCENTAJE_CREDITOS_GRADO'] = (X['CREDITOS_ACUM'] / prom_c).clip(0, 1.5)
        if 'PORCENTAJE_MATERIAS_GRADO' in X.columns:
            X['PORCENTAJE_MATERIAS_GRADO'] = (X['CURSOS_ACUM'] / prom_m).clip(0, 1.5)

    if 'CUMPLE_REGLA_GRADO' in X.columns:
        X['CUMPLE_REGLA_GRADO'] = (
            (X['CURSOS_ACUM'] > UMBRAL_CURSOS) & (X['CREDITOS_ACUM'] > UMBRAL_CREDITOS)
        ).astype(float)

    if con_automata:
        if 'ESTADO_AUTOMATA' in df.columns:
            for c in pd.get_dummies(df['ESTADO_AUTOMATA'], prefix='ESTADO').columns:
                if c in X.columns:
                    X[c] = pd.get_dummies(df['ESTADO_AUTOMATA'], prefix='ESTADO')[c].values
        if 'TRANSICION_AUTOMATA' in df.columns:
            for c in pd.get_dummies(df['TRANSICION_AUTOMATA'], prefix='TRANS').columns:
                if c in X.columns:
                    X[c] = pd.get_dummies(df['TRANSICION_AUTOMATA'], prefix='TRANS')[c].values
        if 'RIESGO_ACADEMICO' in X.columns and 'ESTADO_AUTOMATA' in df.columns:
            X['RIESGO_ACADEMICO'] = df['ESTADO_AUTOMATA'].isin(['PAP', 'PAT', 'PFU']).astype(float)

    pred = model.predict(X)
    idx_to_class = metadata['idx_to_class']
    return pd.Series(pred).map(lambda i: idx_to_class[int(i)])


# =============================================================================
# CARGAR MODELOS Y OPCIONES
# =============================================================================

models = load_models()
model_a, meta_a = models['automata']
model_n, meta_n = models['numerico']
cv_a = meta_a['cv_summary']
cv_n = meta_n['cv_summary']
n_rows = meta_a['n_rows']
n_students = meta_a['n_students']

estado_opts, trans_opts, programa_opts = load_estado_options()

# =============================================================================
# HEADER
# =============================================================================

st.markdown(f"""
<div class="main-header">
    <h1>Simulador de Trayectorias Academicas</h1>
    <p>Modelo Predictivo XGBoost · Universidad Tecnologica de Bolivar</p>
    <span class="badge">Dual: Automata F1={cv_a['f1_macro_mean']:.4f} vs Numerico F1={cv_n['f1_macro_mean']:.4f} · {n_rows:,} filas · {n_students:,} estudiantes</span>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(["Prediccion Individual", "Prediccion Masiva (Batch)", "Metricas del Modelo"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICCION INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    modelo_sel = st.radio(
        "Modelo a utilizar",
        ["Automata (con OHE)", "Numerico (sin leakage)"],
        horizontal=True,
        key="tab1_modelo",
    )
    con_automata = modelo_sel.startswith("Automata")

    if con_automata:
        st.info("**Modelo Automata:** recibe estado actual + transicion del automata como features. "
                "Resultado esperado ~100% porque el automata es determinista.")
    else:
        st.info("**Modelo Numerico:** predice SOLO con datos academicos (PPA, creditos, cursos, "
                "orden, programa). Sin knowledge del automata. Resultados mas realistas.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">Datos del estudiante</div>', unsafe_allow_html=True)

        if con_automata:
            estado = st.selectbox("Estado academico actual", estado_opts, index=2)
            transicion = st.selectbox(
                "Transicion del automata", trans_opts,
                format_func=lambda t: f"{t} -- {TRANS_DESCRIPCIONES.get(t, '')}",
                help="Simbolo de transicion del automata finito",
            )
        else:
            estado = None
            transicion = None

        def _sync_ppa_from_slider():
            st.session_state.ppa_num = st.session_state.ppa_sl

        def _sync_ppa_from_number():
            st.session_state.ppa_sl = st.session_state.ppa_num

        if "ppa_sl" not in st.session_state:
            st.session_state.ppa_sl = 3.50
        if "ppa_num" not in st.session_state:
            st.session_state.ppa_num = 3.50

        st.markdown("**PPA -- Promedio Ponderado Acumulado**")
        col_ppa_sl, col_ppa_num = st.columns([3, 1])
        with col_ppa_sl:
            st.slider(
                "PPA", 0.0, 5.0,
                step=0.01, label_visibility="collapsed",
                key="ppa_sl", on_change=_sync_ppa_from_slider,
            )
        with col_ppa_num:
            st.number_input(
                "PPA", 0.0, 5.0, step=0.01,
                format="%.2f", label_visibility="collapsed",
                key="ppa_num", on_change=_sync_ppa_from_number,
            )
        ppa_val = st.session_state.ppa_sl

        st.markdown("**Avance academico**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            creditos = st.number_input("Creditos aprobados", 0, 500, 120, step=1)
        with col_c2:
            cursos = st.number_input("Materias aprobadas", 0, 200, 30, step=1)

        st.markdown("**Programa y posicion en la trayectoria**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if programa_opts:
                programa = st.selectbox("Programa", programa_opts)
            else:
                programa = st.text_input("Programa", "INGENIERIA DE SISTEMAS")
        with col_p2:
            orden = st.number_input(
                "N.o de periodo en la carrera", 1, 30, 5, step=1,
                help="Posicion secuencial del periodo en la trayectoria",
            )

        st.markdown("")
        predecir = st.button("Predecir siguiente estado", use_container_width=True)

        if con_automata:
            with st.expander("Tabla de transiciones del automata (referencia)", expanded=False):
                ref = pd.DataFrame([
                    {"Simbolo": k, "Significado": v}
                    for k, v in sorted(TRANS_DESCRIPCIONES.items())
                ])
                st.dataframe(ref, use_container_width=True, hide_index=True, height=340)

    with col_result:
        st.markdown('<div class="section-title">Resultado de la prediccion</div>', unsafe_allow_html=True)

        if predecir:
            modelo_obj = model_a if con_automata else model_n
            meta_obj = meta_a if con_automata else meta_n

            estado_siguiente, probabilidades = predict_individual(
                modelo_obj, meta_obj, estado, transicion,
                ppa_val, cursos, creditos, orden, programa,
                con_automata=con_automata,
            )

            color = PALETTE.get(estado_siguiente, "#666")
            bg = BG_PALETTE.get(estado_siguiente, "#F8F9FA")
            best_prob = probabilidades[estado_siguiente]

            if con_automata:
                transition_html = f"{estado} &rarr; <b>{estado_siguiente}</b>"
                trans_desc = f"{transicion} -- {TRANS_DESCRIPCIONES.get(transicion, '')}"
            else:
                transition_html = f"<b>{estado_siguiente}</b>"
                trans_desc = None

            st.markdown(f"""
            <div class="prediction-result-box" style="border-color:{color};background:{bg}">
                <div class="pred-label">El estudiante avanzara a</div>
                <div class="pred-estado" style="color:{color}">{estado_siguiente}</div>
                <div class="pred-conf" style="color:{color}">
                    Confianza del modelo: {best_prob * 100:.1f}%
                </div>
                <div class="pred-trans" style="margin-top:0.8rem;font-size:1.1rem;color:#334155">
                    {transition_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Resumen de la prediccion")
            resumen_data = [("Campo", "Valor")]
            resumen_data.append(("Modelo utilizado", modelo_sel))
            if con_automata:
                resumen_data.append(("Estado actual del estudiante", estado))
                resumen_data.append(("Transicion aplicada", trans_desc))
            resumen_data += [
                ("PPA acumulado", f"{ppa_val:.2f}"),
                ("Creditos aprobados", str(creditos)),
                ("Materias aprobadas", str(cursos)),
                ("Programa", programa),
                ("Periodo en la trayectoria", f"#{orden}"),
                ("", ""),
                ("ESTADO PREDICHO", f"{estado_siguiente}"),
                ("Confianza del modelo", f"{best_prob * 100:.1f}%"),
            ]
            det_df = pd.DataFrame(resumen_data[1:], columns=resumen_data[0])
            st.dataframe(det_df, use_container_width=True, hide_index=True, height=380)

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            with m1:
                pct_cred = (creditos / PROM_CREDITOS_GRADO.get(programa, 15)) * 100
                st.metric("Avance en creditos", f"{pct_cred:.1f}%")
            with m2:
                nivel = {
                    "Continuo regular": "Bajo", "Reingreso": "Bajo",
                    "Reinicio": "Bajo", "Transferencia interna": "Bajo",
                    "PAP": "Medio", "Recuperacion academica": "Medio",
                    "PAT": "Alto", "PFU": "Alto",
                    "Final": "Ciritico", "Exclusion": "Ciritico",
                }.get(estado_siguiente, "--")
                st.metric("Nivel de riesgo", nivel)
            with m3:
                rec = {
                    "Grado": "Verificar tramite", "Continuo regular": "Mantener ritmo",
                    "PAP": "Apoyo academico", "PAT": "Intervencion urgente",
                    "Recuperacion academica": "Tutoria", "PFU": "Contactar estudiante",
                    "Final": "Proceso administrativo", "Exclusion": "Proceso administrativo",
                }.get(estado_siguiente, "--")
                st.metric("Accion recomendada", rec)

            st.markdown("---")
            st.markdown("#### Probabilidades por clase")

            proba_df = pd.DataFrame([
                {"Estado": k, "Probabilidad": v, "Porcentaje": f"{v*100:.1f}%"}
                for k, v in sorted(probabilidades.items(), key=lambda x: -x[1])
            ])
            proba_df['Color'] = proba_df['Estado'].map(lambda x: PALETTE.get(x, "#888"))
            proba_df['Ranking'] = range(1, len(proba_df) + 1)
            proba_df_show = proba_df[['Ranking', 'Estado', 'Probabilidad', 'Porcentaje']].copy()
            proba_df_show['Probabilidad'] = proba_df_show['Probabilidad'].map(lambda x: f"{x:.4f}")
            st.dataframe(proba_df_show, use_container_width=True, hide_index=True, height=340)

            fig_p, ax_p = plt.subplots(figsize=(8, 4))
            ax_p.barh(
                proba_df['Estado'], proba_df['Probabilidad'],
                color=proba_df['Color'].tolist(), height=0.6, edgecolor='white',
            )
            ax_p.set_xlim(0, 1.05)
            ax_p.set_xlabel('Probabilidad')
            ax_p.set_title('Distribucion de probabilidad del modelo', fontweight='bold')
            ax_p.invert_yaxis()
            for s in ['top', 'right']:
                ax_p.spines[s].set_visible(False)
            for i, v in enumerate(proba_df['Probabilidad']):
                ax_p.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=8)
            fig_p.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close(fig_p)
        else:
            st.markdown("""
            <div style="border:2px dashed #CBD5E1;border-radius:14px;padding:3rem 2rem;
            text-align:center;color:#94A3B8;margin-top:0.5rem">
                <div style="font-size:1rem;font-weight:500">
                    Complete los datos del estudiante y presione<br>
                    <strong>Predecir siguiente estado</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PREDICCION MASIVA (BATCH)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    modelo_sel_b = st.radio(
        "Modelo a utilizar",
        ["Automata (con OHE)", "Numerico (sin leakage)"],
        horizontal=True,
        key="tab2_modelo",
    )
    con_automata_b = modelo_sel_b.startswith("Automata")

    if con_automata_b:
        st.markdown("""
        <div style="background:#F0F7FF;border:1px solid #BFDBFE;border-radius:12px;
        padding:1rem 1.3rem;margin-bottom:1.3rem;font-size:0.88rem;color:#1E40AF">
        <strong>Formato del CSV/Excel (Modelo Automata):</strong> debe contener las columnas
        <code>ESTADO_AUTOMATA</code>, <code>TRANSICION_AUTOMATA</code>,
        <code>PROMEDIO_ACUMULADO</code>, <code>CREDITOS_APROVADOS</code>,
        <code>NRO_CURSOS_APROBADOS</code>.
        Opcional: <code>PROGRAMA</code>, <code>PERIODO</code>, <code>ID</code>, <code>ESTADO_CONSECUENTE</code>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#EDFAF4;border:1px solid #A7F3D0;border-radius:12px;
        padding:1rem 1.3rem;margin-bottom:1.3rem;font-size:0.88rem;color:#065F46">
        <strong>Formato del CSV/Excel (Modelo Numerico):</strong> solo necesita columnas academicas:
        <code>PROMEDIO_ACUMULADO</code>, <code>CREDITOS_APROVADOS</code>,
        <code>NRO_CURSOS_APROBADOS</code>.
        Opcional: <code>PROGRAMA</code>, <code>PERIODO</code>, <code>ID</code>, <code>ESTADO_CONSECUENTE</code>.
        NO necesita <code>ESTADO_AUTOMATA</code> ni <code>TRANSICION_AUTOMATA</code>.
        </div>
        """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Arrastra o selecciona tu CSV/Excel", type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        with st.spinner("Leyendo archivo..."):
            try:
                df_up = pd.read_excel(uploaded)
            except Exception:
                try:
                    df_up = pd.read_csv(uploaded, encoding="utf-8-sig")
                except Exception:
                    df_up = pd.read_csv(uploaded, encoding="latin-1")

        required_base = {'PROMEDIO_ACUMULADO', 'CREDITOS_APROVADOS', 'NRO_CURSOS_APROBADOS'}
        if con_automata_b:
            required_base |= {'ESTADO_AUTOMATA', 'TRANSICION_AUTOMATA'}
        missing = required_base - set(df_up.columns)
        if missing:
            st.error(f"Faltan columnas: {', '.join(missing)}")
            st.stop()

        with st.spinner("Generando predicciones..."):
            modelo_obj = model_a if con_automata_b else model_n
            meta_obj = meta_a if con_automata_b else meta_n
            preds = predict_batch(modelo_obj, meta_obj, df_up, con_automata=con_automata_b)

        df_up['PREDICCION_XGB'] = preds.values
        has_target = 'ESTADO_CONSECUENTE' in df_up.columns

        st.markdown(f'<div class="section-title">Resultados -- {modelo_sel_b}</div>', unsafe_allow_html=True)

        if has_target:
            df_eval = df_up[df_up['PREDICCION_XGB'].notna() & df_up['ESTADO_CONSECUENTE'].notna()].copy()
            y_true = df_eval['ESTADO_CONSECUENTE'].astype(str)
            y_pred = df_eval['PREDICCION_XGB'].astype(str)
            acc = accuracy_score(y_true, y_pred)
            f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
            n_eval = len(df_eval)
            n_correct = int((y_true == y_pred).sum())
            n_wrong = n_eval - n_correct

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.markdown(
                    f"""<div class="metric-card" style="background:#EBF3FC;border-color:#BFDBFE">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value" style="color:#185FA5">{acc*100:.2f}%</div>
                    <div class="metric-sub">{n_correct:,} correctos de {n_eval:,}</div></div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(
                    f"""<div class="metric-card" style="background:#EDFAF4;border-color:#A7F3D0">
                    <div class="metric-label">F1 Macro</div>
                    <div class="metric-value" style="color:#1D9E75">{f1m:.4f}</div>
                    <div class="metric-sub">todas las clases iguales</div></div>""", unsafe_allow_html=True)
            with mc3:
                st.markdown(
                    f"""<div class="metric-card" style="background:#EDFAF4;border-color:#A7F3D0">
                    <div class="metric-label">F1 Weighted</div>
                    <div class="metric-value" style="color:#1D9E75">{f1w:.4f}</div>
                    <div class="metric-sub">ponderado por soporte</div></div>""", unsafe_allow_html=True)
            with mc4:
                st.markdown(
                    f"""<div class="metric-card" style="background:#FEF6E8;border-color:#FDE68A">
                    <div class="metric-label">Casos Incorrectos</div>
                    <div class="metric-value" style="color:#D85A30">{n_wrong:,}</div>
                    <div class="metric-sub">{n_wrong/n_eval*100:.2f}% del total</div></div>""", unsafe_allow_html=True)
            with mc5:
                st.markdown(
                    f"""<div class="metric-card" style="background:#F3F3F2;border-color:#D1D5DB">
                    <div class="metric-label">Total Evaluados</div>
                    <div class="metric-value" style="color:#64748B">{n_eval:,}</div>
                    <div class="metric-sub">con target conocido</div></div>""", unsafe_allow_html=True)

            st.markdown("---")
            col_rep, col_cm = st.columns([1, 1.6], gap="large")

            with col_rep:
                st.markdown('<div class="section-title">Reporte por clase (Precision / Recall / F1)</div>', unsafe_allow_html=True)
                rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
                clases = [k for k in rep if k not in ("accuracy", "macro avg", "weighted avg")]
                rep_df = pd.DataFrame({
                    "Estado": clases,
                    "Precision": [f"{rep[c]['precision']:.4f}" for c in clases],
                    "Recall": [f"{rep[c]['recall']:.4f}" for c in clases],
                    "F1": [f"{rep[c]['f1-score']:.4f}" for c in clases],
                    "Soporte": [f"{int(rep[c]['support']):,}" for c in clases],
                })
                st.dataframe(rep_df, use_container_width=True, hide_index=True,
                             height=min(38 * (len(clases) + 1), 480))

                fig_f1, ax_f1 = plt.subplots(figsize=(5, 3.5))
                f1_vals = [rep[c]["f1-score"] for c in clases]
                ax_f1.barh(clases, f1_vals, color=[PALETTE.get(c, "#888") for c in clases],
                           height=0.6, edgecolor="white")
                ax_f1.set_xlim(0, 1.08)
                ax_f1.set_xlabel("F1 Score", fontsize=9)
                ax_f1.axvline(x=1.0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
                for s in ["top", "right"]:
                    ax_f1.spines[s].set_visible(False)
                for i, v in enumerate(f1_vals):
                    ax_f1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7.5)
                fig_f1.tight_layout()
                st.pyplot(fig_f1, use_container_width=True)
                plt.close(fig_f1)

            with col_cm:
                st.markdown('<div class="section-title">Matriz de confusion</div>', unsafe_allow_html=True)
                all_cls = sorted(y_true.unique())
                cm = confusion_matrix(y_true, y_pred, labels=all_cls)
                cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

                vista = st.radio("Vista", ["Proporciones (0-1)", "Conteos absolutos"],
                                 horizontal=True, label_visibility="collapsed")
                data_cm = cm_norm if vista == "Proporciones (0-1)" else cm
                fmt_cm = ".2f" if vista == "Proporciones (0-1)" else "d"

                fig_cm, ax_cm = plt.subplots(
                    figsize=(max(7, len(all_cls) * 0.85), max(6, len(all_cls) * 0.78)))
                sns.heatmap(data_cm, annot=True, fmt=fmt_cm, cmap="Blues",
                            xticklabels=all_cls, yticklabels=all_cls, ax=ax_cm,
                            linewidths=0.5, linecolor="white", annot_kws={"size": 8})
                ax_cm.set_xlabel("Estado predicho", fontsize=10)
                ax_cm.set_ylabel("Estado real", fontsize=10)
                ax_cm.set_title(f"Matriz de Confusion -- {acc*100:.2f}% accuracy",
                                fontsize=11, fontweight="bold")
                ax_cm.tick_params(axis="x", rotation=45, labelsize=8)
                fig_cm.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)

            st.markdown("---")
        else:
            n_total = len(df_up)
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(
                    f"""<div class="metric-card" style="background:#EBF3FC;border-color:#BFDBFE">
                    <div class="metric-label">Filas procesadas</div>
                    <div class="metric-value" style="color:#185FA5">{n_total:,}</div></div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(
                    f"""<div class="metric-card" style="background:#EDFAF4;border-color:#A7F3D0">
                    <div class="metric-label">Predicciones</div>
                    <div class="metric-value" style="color:#1D9E75">{df_up['PREDICCION_XGB'].notna().sum():,}</div></div>""", unsafe_allow_html=True)
            st.info("Sin columna ESTADO_CONSECUENTE -- solo se muestran predicciones.")

        st.markdown('<div class="section-title">Distribucion de predicciones</div>', unsafe_allow_html=True)
        dist = df_up['PREDICCION_XGB'].dropna().value_counts().reset_index()
        dist.columns = ['Estado', 'Registros']
        dist['Color'] = dist['Estado'].map(lambda x: PALETTE.get(x, "#888"))

        fig_d, ax_d = plt.subplots(figsize=(12, 3.8))
        bars = ax_d.bar(dist['Estado'], dist['Registros'], color=dist['Color'].tolist(),
                        width=0.7, edgecolor='white')
        ax_d.set_ylabel('Registros')
        ax_d.set_title('Distribucion del Estado Predicho', fontsize=12, fontweight='bold')
        ax_d.bar_label(bars, fmt="%d", padding=3, fontsize=9)
        ax_d.tick_params(axis='x', rotation=35, labelsize=9)
        for s in ['top', 'right']:
            ax_d.spines[s].set_visible(False)
        fig_d.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close(fig_d)

        st.markdown('<div class="section-title">Muestra</div>', unsafe_allow_html=True)
        show_cols = [c for c in ['ID', 'PERIODO', 'PROGRAMA', 'ESTADO_AUTOMATA',
                                 'TRANSICION_AUTOMATA', 'ESTADO_CONSECUENTE', 'PREDICCION_XGB']
                     if c in df_up.columns]
        st.dataframe(df_up[show_cols].head(100), use_container_width=True, hide_index=True, height=320)
        if len(df_up) > 100:
            st.caption(f"Mostrando 100 de {len(df_up):,} filas.")

        st.markdown('<div class="section-title">Descargar</div>', unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_bytes = df_up.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("Descargar CSV", data=csv_bytes,
                               file_name="predicciones_xgb.csv", mime="text/csv",
                               use_container_width=True)
        with col_dl2:
            try:
                import openpyxl
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    df_up.to_excel(w, index=False, sheet_name="Predicciones")
                buf.seek(0)
                st.download_button("Descargar Excel", data=buf.getvalue(),
                                   file_name="predicciones_xgb.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            except ImportError:
                st.info("pip install openpyxl para Excel")
    else:
        st.markdown("""
        <div style="border:2px dashed #CBD5E1;border-radius:14px;padding:3.5rem 2rem;
        text-align:center;color:#94A3B8;margin-top:0.5rem">
            <div style="font-size:1rem;font-weight:500;margin-bottom:0.5rem">
                Arrastra tu CSV/Excel aqui o seleccionalo con el boton
            </div>
            <div style="font-size:0.85rem">
                Columnas requeridas: <code>PROMEDIO_ACUMULADO</code>,
                <code>CREDITOS_APROVADOS</code>, <code>NRO_CURSOS_APROBADOS</code>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — METRICAS DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="background:#FFF7ED;border:1px solid #FDBA74;border-radius:12px;
    padding:1rem 1.3rem;margin-bottom:1.3rem;font-size:0.88rem;color:#9A3412">
    <strong>Comparacion de modelos:</strong> el modelo <em>Automata</em> recibe OHE de
    <code>ESTADO_AUTOMATA</code> + <code>TRANSICION_AUTOMATA</code> como features, lo cual
    equivale a una tabla de lookup del automata determinista (resultados ~100%).
    El modelo <em>Numerico</em> predice SOLO con datos academicos (PPA, creditos, cursos),
    sin knowledge del automata — resultados mas realistas y representativos.
    </div>
    """, unsafe_allow_html=True)

    # ── Comparacion lado a lado ──
    st.markdown('<div class="section-title">Comparacion lado a lado</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:#FEF2F2;border:2px solid #EF4444;border-radius:12px;padding:1.2rem;text-align:center">
            <div style="font-size:0.75rem;text-transform:uppercase;font-weight:700;color:#991B1B;letter-spacing:0.05em">
                Modelo AUTOMATA (con OHE)
            </div>
            <div style="font-size:2.2rem;font-weight:900;color:#DC2626;margin:0.3rem 0">
                {cv_a['f1_macro_mean']:.4f}
            </div>
            <div style="font-size:0.8rem;color:#991B1B">
                F1-Macro +/- {cv_a['f1_macro_std']:.4f}<br>
                Accuracy: {cv_a['accuracy_mean']:.4f}<br>
                Features: {len(meta_a['feature_names'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:#F0FDF4;border:2px solid #22C55E;border-radius:12px;padding:1.2rem;text-align:center">
            <div style="font-size:0.75rem;text-transform:uppercase;font-weight:700;color:#166534;letter-spacing:0.05em">
                Modelo NUMERICO (sin leakage)
            </div>
            <div style="font-size:2.2rem;font-weight:900;color:#16A34A;margin:0.3rem 0">
                {cv_n['f1_macro_mean']:.4f}
            </div>
            <div style="font-size:0.8rem;color:#166534">
                F1-Macro +/- {cv_n['f1_macro_std']:.4f}<br>
                Accuracy: {cv_n['accuracy_mean']:.4f}<br>
                Features: {len(meta_n['feature_names'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    gap = cv_a['f1_macro_mean'] - cv_n['f1_macro_mean']
    st.markdown(f"""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;
    padding:0.8rem 1.2rem;margin:0.8rem 0;font-size:0.9rem;text-align:center">
    <strong>Gap:</strong> {gap:.4f} — representa la informacion que aporta el
    conocimiento del automata (estado + transicion) vs. solo datos academicos.
    </div>
    """, unsafe_allow_html=True)

    # ── Explicacion del data leakage ──
    with st.expander("Por que el modelo Automata da ~100%? (Explicacion del Data Leakage)", expanded=False):
        st.markdown("""
        El automata finito utilizado es **determinista**: cada par `(ESTADO_AUTOMATA, TRANSICION_AUTOMATA)`
        mapea a exactamente un `ESTADO_CONSECUENTE`.

        Cuando el modelo recibe `OHE(ESTADO_AUTOMATA)` + `OHE(TRANSICION_AUTOMATA)` como features,
        es como si tuviera la **tabla de lookup** del automata como input. No necesita "aprender" nada —
        simplemente memoriza esta tabla.

        **Verificacion:** 25 de 26 pares (96.2%) son 100% deterministicos. La unica excepcion es
        `(Reinicio, j)` que puede llevar a "Primera vez en una carrera" (557) o "Continuo regular" (30).

        Esto NO es overfitting — es **data leakage por disenio**. Las features contienen la respuesta.
        El modelo Numerico resuelve esto al predecir solo con datos academicos reales.
        """)

    tab_auto, tab_num = st.tabs(["Modelo Automata", "Modelo Numerico"])

    for label, meta_tab, dir_out in [
        ("Automata", meta_a, 'outputs'),
        ("Numerico", meta_n, 'outputs/numerico'),
    ]:
        tab = tab_auto if label == "Automata" else tab_num
        cv_tab = meta_tab['cv_summary']

        with tab:
            st.markdown(f'<div class="section-title">Metricas consolidadas (5-Fold CV) — {label}</div>',
                        unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    f"""<div class="metric-card" style="background:#EBF3FC;border-color:#BFDBFE">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value" style="color:#185FA5">{cv_tab['accuracy_mean']:.4f}</div>
                    <div class="metric-sub">+/- {cv_tab['accuracy_std']:.4f}</div></div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(
                    f"""<div class="metric-card" style="background:#EDFAF4;border-color:#A7F3D0">
                    <div class="metric-label">F1 Macro</div>
                    <div class="metric-value" style="color:#1D9E75">{cv_tab['f1_macro_mean']:.4f}</div>
                    <div class="metric-sub">+/- {cv_tab['f1_macro_std']:.4f}</div></div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(
                    f"""<div class="metric-card" style="background:#EDFAF4;border-color:#A7F3D0">
                    <div class="metric-label">F1 Weighted</div>
                    <div class="metric-value" style="color:#1D9E75">{cv_tab['f1_weighted_mean']:.4f}</div>
                    <div class="metric-sub">+/- {cv_tab['f1_weighted_std']:.4f}</div></div>""", unsafe_allow_html=True)
            with k4:
                st.markdown(
                    f"""<div class="metric-card" style="background:#FEF6E8;border-color:#FDE68A">
                    <div class="metric-label">Features</div>
                    <div class="metric-value" style="color:#D85A30">{len(meta_tab['feature_names'])}</div>
                    <div class="metric-sub">{n_rows:,} filas</div></div>""", unsafe_allow_html=True)

            fold_metrics = meta_tab.get('fold_metrics', [])
            if fold_metrics:
                st.markdown(f'<div class="section-title">Metricas por fold</div>', unsafe_allow_html=True)
                fold_df = pd.DataFrame(fold_metrics)
                fold_df.columns = ['Fold', 'Accuracy', 'F1 Macro', 'F1 Weighted']
                fold_df['Accuracy'] = fold_df['Accuracy'].map(lambda x: f"{x:.4f}")
                fold_df['F1 Macro'] = fold_df['F1 Macro'].map(lambda x: f"{x:.4f}")
                fold_df['F1 Weighted'] = fold_df['F1 Weighted'].map(lambda x: f"{x:.4f}")
                fold_df.loc[len(fold_df)] = [
                    "Promedio",
                    f"{cv_tab['accuracy_mean']:.4f}",
                    f"{cv_tab['f1_macro_mean']:.4f}",
                    f"{cv_tab['f1_weighted_mean']:.4f}",
                ]
                st.dataframe(fold_df, use_container_width=True, hide_index=True)

                folds_plot = [f for f in fold_metrics if isinstance(f, dict) and 'accuracy' in f]
                if folds_plot:
                    fig_folds, ax_folds = plt.subplots(figsize=(8, 3.5))
                    fold_nums = [f['fold'] for f in folds_plot]
                    acc_vals = [f['accuracy'] for f in folds_plot]
                    f1m_vals = [f['f1_macro'] for f in folds_plot]
                    ax_folds.plot(fold_nums, acc_vals, 'o-', color='#185FA5', label='Accuracy', linewidth=2)
                    ax_folds.plot(fold_nums, f1m_vals, 's-', color='#1D9E75', label='F1 Macro', linewidth=2)
                    ax_folds.set_xlabel('Fold')
                    ax_folds.set_ylabel('Score')
                    ax_folds.set_title(f'Metricas por Fold — {label}', fontweight='bold')
                    ax_folds.set_xticks(fold_nums)
                    ax_folds.set_ylim(min(min(acc_vals), min(f1m_vals)) - 0.002, 1.002)
                    ax_folds.legend()
                    ax_folds.grid(True, alpha=0.3)
                    for s in ['top', 'right']:
                        ax_folds.spines[s].set_visible(False)
                    fig_folds.tight_layout()
                    st.pyplot(fig_folds, use_container_width=True)
                    plt.close(fig_folds)

            fi_path = os.path.join(dir_out, 'feature_importance.png')
            if os.path.exists(fi_path):
                st.markdown(f'<div class="section-title">Importancia de features</div>', unsafe_allow_html=True)
                st.image(fi_path, use_container_width=True)

            cm_path = os.path.join(dir_out, 'confusion_matrix.png')
            if os.path.exists(cm_path):
                st.markdown(f'<div class="section-title">Matriz de confusion — ultimo fold</div>', unsafe_allow_html=True)
                st.image(cm_path, use_container_width=True)

            cr_path = os.path.join(dir_out, 'classification_report.json')
            if os.path.exists(cr_path):
                st.markdown(f'<div class="section-title">Classification report — ultimo fold</div>', unsafe_allow_html=True)
                with open(cr_path, 'r', encoding='utf-8') as f:
                    cr = json.load(f)
                clases_cr = [k for k in cr if k not in ("accuracy", "macro avg", "weighted avg")]
                if clases_cr:
                    cr_df = pd.DataFrame({
                        "Clase": clases_cr,
                        "Precision": [f"{cr[c]['precision']:.4f}" for c in clases_cr],
                        "Recall": [f"{cr[c]['recall']:.4f}" for c in clases_cr],
                        "F1-Score": [f"{cr[c]['f1-score']:.4f}" for c in clases_cr],
                        "Soporte": [f"{int(cr[c]['support']):,}" for c in clases_cr],
                    })
                    summary_rows = []
                    for avg_key, avg_label in [("macro avg", "Macro Promedio"), ("weighted avg", "Weighted Promedio")]:
                        if avg_key in cr:
                            summary_rows.append({
                                "Clase": avg_label,
                                "Precision": f"{cr[avg_key]['precision']:.4f}",
                                "Recall": f"{cr[avg_key]['recall']:.4f}",
                                "F1-Score": f"{cr[avg_key]['f1-score']:.4f}",
                                "Soporte": f"{int(cr[avg_key]['support']):,}",
                            })
                    cr_df = pd.concat([cr_df, pd.DataFrame(summary_rows)], ignore_index=True)
                    st.dataframe(cr_df, use_container_width=True, hide_index=True)
                    if 'accuracy' in cr:
                        st.markdown(f"**Accuracy global del fold:** {cr['accuracy']:.4f}")

                    fig_cr, ax_cr = plt.subplots(figsize=(8, 4))
                    f1_vals = [cr[c]['f1-score'] for c in clases_cr]
                    ax_cr.barh(clases_cr, f1_vals,
                               color=[PALETTE.get(c, '#888') for c in clases_cr],
                               height=0.6, edgecolor='white')
                    ax_cr.set_xlim(0, 1.08)
                    ax_cr.set_xlabel('F1-Score')
                    ax_cr.set_title(f'F1-Score por clase — {label}', fontweight='bold')
                    ax_cr.invert_yaxis()
                    for s in ['top', 'right']:
                        ax_cr.spines[s].set_visible(False)
                    for i, v in enumerate(f1_vals):
                        ax_cr.text(v + 0.005, i, f"{v:.4f}", va='center', fontsize=8)
                    fig_cr.tight_layout()
                    st.pyplot(fig_cr, use_container_width=True)
                    plt.close(fig_cr)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Sobre los modelos")
    st.markdown(f"""
    **Modelo Predictivo de Trayectorias Academicas**

    Dos modelos XGBoost que predicen el estado academico siguiente.

    ---
    **Automata (con OHE)**
    - F1-Macro: {cv_a['f1_macro_mean']:.4f} +/- {cv_a['f1_macro_std']:.4f}
    - Accuracy: {cv_a['accuracy_mean']:.4f}
    - Features: {len(meta_a['feature_names'])}

    **Numerico (sin leakage)**
    - F1-Macro: {cv_n['f1_macro_mean']:.4f} +/- {cv_n['f1_macro_std']:.4f}
    - Accuracy: {cv_n['accuracy_mean']:.4f}
    - Features: {len(meta_n['feature_names'])}

    - **Filas:** {n_rows:,}
    - **Estudiantes:** {n_students:,}
    """)

    st.markdown("---")
    st.markdown("""
    **Pasos:**
    1. `py -3.10 entrenar.py` — entrena ambos modelos
    2. `py -3.10 -m streamlit run app.py` — ejecuta la app

    ---
    Universidad Tecnologica de Bolivar
    """)
