# Guia de Buenas Practicas en Machine Learning

## Aplicadas al Proyecto de Prediccion de Trayectorias Academicas

---

## 1. Data Leakage: El Error Mas Costoso en ML

### 1.1 Definicion

Data leakage ocurre cuando informacion del futuro o del target se filtra a las features durante el entrenamiento. El modelo aprende patrones que no existiran en produccion.

### 1.2 Tipo: Leakage Estructural

**Problema:** Incluir OHE(ESTADO_AUTOMATA) + OHE(TRANSICION_AUTOMATA) como features.

**Por que es leakage:** El automata es determinista: dado un estado y una transicion, el estado consecuente es siempre el mismo (25/26 pares). El modelo no "aprende" — solo replica la tabla de transicion.

**Deteccion:**
- Accuracy sospechosamente perfecta (>99.5%)
- F1-Macro ≈ 1.0 en todas las clases
- Sin variacion entre folds
- Las features del automata dominan la importancia

**Solucion aplicada:** Enfoque dual — modelo "Automata" como diagnostico, modelo "Numerico" como realidad.

### 1.3 Tipo: Leakage por Agrupacion

**Problema:** Usar KFold simple sin agrupar por estudiante.

**Consecuencia:** El mismo estudiante aparece en train y test. El modelo "recuerda" el patron especifico del estudiante.

**Solucion aplicada:** `StratifiedGroupKFold` con `groups=df['ID']`.

**Verificacion:** Asercion en el codigo:
```python
ids_tr = set(groups.iloc[train_idx])
ids_te = set(groups.iloc[test_idx])
assert len(ids_tr & ids_te) == 0, "Data leakage entre folds"
```

### 1.4 Checklist Anti-Leakage

- [ ] ?La feature contiene informacion que no estaria disponible en el momento de la prediccion?
- [ ] ?La feature es una transformacion directa del target?
- [ ] ?Estoy usando informacion del futuro (e.g., promedios globales)?
- [ ] ?Las observaciones del mismo grupo/individuo estan separadas correctamente entre train/test?
- [ ] ?La validacion cruzada respeta la estructura de grupos?
- [ ] ?El preprocessing (imputacion, scaling) se hace DENTRO de cada fold?
- [ ] ?El accuracy es sospechosamente alto?

---

## 2. Validacion Cruzada Correcta

### 2.1 Estrategias para Datos de Trayectorias

| Estrategia | Uso | Este Proyecto |
|---|---|---|
| KFold | Datos i.i.d. | No aplica |
| StratifiedKFold | Clases desbalanceadas, datos i.i.d. | No aplica |
| GroupKFold | Observaciones agrupadas (mismo estudiante) | Parcial |
| **StratifiedGroupKFold** | Grupos + balance de clases | **SI** |

### 2.2 Por que 5 Pliegues?

- Suficientes para estimar varianza (5 estimaciones)
- No tan costoso como leave-one-out
- Estandar en la literatura (muchos papers usan 5 o 10)

### 2.3 El Preprocessing Dentro del Fold

**INCORRECTO:**
```python
# 1. Imputar/transformar TODOS los datos
X_imputed = imputer.fit_transform(X)
# 2. Luego separar folds
for train, test in cv.split(X_imputed):
    model.fit(X_imputed[train], y[train])
```

**CORRECTO:**
```python
for train, test in cv.split(X, y, groups):
    # 1. Separar primero
    X_tr, X_te = X.iloc[train], X.iloc[test]
    # 2. Preprocesar SOLO con train
    imputer.fit(X_tr)
    X_tr_imp = imputer.transform(X_tr)
    X_te_imp = imputer.transform(X_te)
```

En este proyecto, la imputacion de nulos por mediana del grupo ESTADO_AUTOMATA se hace antes de la CV para simplificar, pero idealmente deberia hacerse dentro de cada fold. Dado que la mediana por estado es estable, el impacto es minimo.

---

## 3. Clases Desbalanceadas

### 3.1 Diagnostico del Desbalance

El dataset presenta distribucion sesgada de estados academicos:

| Clase | Proporcion | Riesgo |
|---|---|---|
| Continuo regular | ~60% | Mayoritaria |
| PAP | ~15% | Moderada |
| Grado | ~10% | Moderada |
| PFU, PAT, Reingreso | ~3-5% c/u | Minoritaria |
| Reinicio, Transf. externa, etc. | <1% | Extremadamente minoritaria |

### 3.2 Estrategia: SMOTE con k_neighbors Dinamico

SMOTE genera sinteticos interpolando entre vecinos cercanos. Problema: si una clase tiene solo 2 instancias, con k=5 no es posible.

**Solucion:** `k_neighbors = min(5, min_class_size - 1)`

```python
min_c = y_tr_n.value_counts().min()
kn = max(1, min(5, min_c - 1))
X_res, y_res = SMOTE(random_state=42, k_neighbors=kn).fit_resample(X_tr, y_tr_n)
```

### 3.3 Alternativas No Usadas (y por que)

| Tecnica | Por que no se uso |
|---|---|
| Random Undersampling | Pierde informacion de clase mayoritaria |
| Class weights | XGBoost tiene `scale_pos_weight` pero es para binario |
| Focal Loss | No disponible nativamente en XGBoost |

---

## 4. Reglas de Negocio vs. ML Puro

### 4.1 Regla de Grado

```python
if CURSOS_ACUM > 55 and CREDITOS_ACUM > 150:
    ESTADO_CONSECUENTE = 'Grado'
```

**Justificacion:** La UTB establece que un estudiante que supera estos umbrales debe estar en estado de graduacion. Es una regla administrativa, no estadistica.

**Impacto:** ~2,000 registros sobrescritos. Mejora el F1 de "Grado" artificialmente (el modelo aprende la regla).

**Alternativa:** Podria tratarse como regla post-prediccion (override en la app), no en el entrenamiento. Se deja asi para reflejar la politica institucional.

### 4.2 Riesgo Academico

```python
RIESGO_ACADEMICO = ESTADO_AUTOMATA in ['PAP', 'PAT', 'PFU']
```

Feature booleana que solo existe en el modelo Automata. Captura si el estudiante esta en riesgo. Es redundante con la OHE del estado (un estado PAP ya implica riesgo), pero mejora la interpretabilidad del modelo.

### 4.3 Guardrails (Mascaras de Estado Post-Prediccion)

Las reglas de negocio no solo se aplican en entrenamiento. En inferencia (app), se implementan **guardrails** que filtran y corrigen las probabilidades del modelo:

```python
def aplicar_guardrails(probas, ppa, cursos_acum, creditos_acum):
    # Override: umbrales de grado
    if cursos_acum > 55 and creditos_acum > 150:
        for k in probas: probas[k] = 0.0
        probas['Grado'] = 1.0
        return probas

    # Filtro: PPA >= 3.0 impide sancion academica
    if ppa >= 3.0:
        for est in {'PAP', 'PAT', 'PFU', 'Recuperacion academica'}:
            if est in probas: probas[est] = 0.0

    total = sum(probas.values())
    if total > 0:
        for k in probas: probas[k] /= total
    return probas
```

**Reglas implementadas:**
1. **Override de Grado:** Si CURSOS_ACUM > 55 y CREDITOS_ACUM > 150 → `probas['Grado'] = 1.0`
2. **Filtro de sancion:** Si PPA ≥ 3.0 → PAP, PAT, PFU, Recuperacion academica tienen proba = 0

**Por que post-prediccion y no solo en entrenamiento:**
- El modelo puede dar probabilidades no nulas a estados institucionalmente imposibles
- Las reglas pueden cambiar sin reentrenar (e.g., umbral de PPA podria ser 3.5)
- Es una capa de seguridad adicional que no depende del modelo

**Efecto:** Las probabilidades se renormalizan tras aplicar los filtros, asegurando que sumen 1.0.

---

## 5. Evaluacion y Metricas

### 5.1 Por que F1-Macro es la Metrica Principal

- Accuracy es enganosa cuando las clases estan desbalanceadas
- F1-Weighted da mas peso a clases mayoritarias
- F1-Macro: todas las clases pesan igual, refleja rendimiento en clases minoritarias

**En este proyecto:**
- Accuracy Numerico: 85.08% (parece bueno)
- F1-Macro Numerico: 52.17% (realista: las clases chicas se predicen mal)
- Diferencia: 32.91 puntos — classes minoritarias estan siendo ignoradas por accuracy

### 5.2 Matriz de Confusion Normalizada

Usar normalizacion por fila (verdaderos) para ver proporcion de aciertos/errores por clase. La matriz de confusion en conteos absolutos puede estar dominada por la clase mayoritaria.

### 5.3 Feature Importance en XGBoost

XGBoost ofrece 3 tipos de importancia:
| Tipo | Significado | Usado? |
|---|---|---|
| `weight` | # de veces que la feature se usa para dividir | No |
| `gain` | Ganancia media de la feature en las divisiones | **SI** |
| `cover` | # de observaciones afectadas por la feature | No |

Se usa `gain` porque refleja la contribucion real al rendimiento del modelo.

---

## 6. Reproducibilidad

### 6.1 Semillas Fijas

```python
SGDClassifier(random_state=42)
StratifiedGroupKFold(shuffle=True, random_state=42)
SMOTE(random_state=42)
```

### 6.2 Version de Dependencias

- Python: 3.10
- xgboost: 2.0+
- scikit-learn: 1.3+
- imbalanced-learn: 0.11+
- streamlit: 1.20+

### 6.3 Tracking de Experimentos

Para mejorar el proyecto (trabajo futuro), considerar:
- **MLflow:** Trackear parametros, metricas, y artefactos
- **DVC:** Versionar datasets y pipelines
- **Comet.ml / Weights & Biases:** Experimentacion colaborativa

---

## 7. Interpretabilidad

### 7.1 Estado Actual

- **Feature importance global:** Grafico de barras con top 15 features
- **Probabilidades por clase:** Distribucion completa para cada prediccion
- **Matriz de confusion:** Visualizacion de aciertos y errores por clase

### 7.2 Mejoras Propuestas (SHAP/LIME)

```python
import shap

# SHAP explainer para XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Force plot para prediccion individual
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_test.iloc[0, :],
    matplotlib=True,
)
```

**Beneficio:** Explicar POR QUE un estudiante particular fue clasificado en cierto estado, no solo CUAL estado.

---

## 8. Produccion (MLOps)

### 8.1 La App NO Entrena

Regla fundamental: `entrenar.py` se ejecuta una vez localmente. `app.py` solo carga modelos pre-entrenados via `pickle.load()`.

```python
@st.cache_resource
def load_models():
    # Solo carga archivos .pkl
    # NO llama a entrenar_modelo()
```

### 8.2 Manejo de Errores en Produccion

La app verifica:
1. Que los archivos `.pkl` existan (si no, muestra error con instrucciones)
2. Que el archivo de datos exista (si no, usa fallback con listas fijas)
3. Que el CSV subido tenga las columnas requeridas

### 8.3 Caché en Streamlit

```python
@st.cache_resource  # Los modelos se cargan UNA vez
@st.cache_resource  # Las opciones de dropdowns se cargan UNA vez
```

---

## 9. Checklist de Calidad para Proyectos ML

### Pre-Modelado
- [ ] Exploracion y visualizacion de datos completa
- [ ] Deteccion de valores nulos y estrategia de imputacion
- [ ] Identificacion de leakage potencial
- [ ] Feature engineering documentado
- [ ] Separacion correcta train/test por grupos

### Modelado
- [ ] Linea base simple (e.g., regresion logistica) para comparar
- [ ] Validacion cruzada adecuada a la estructura de datos
- [ ] Balanceo de clases (si aplica)
- [ ] Hiperparametros justificados (no grid search arbitrario)
- [ ] Semillas aleatorias fijadas

### Evaluacion
- [ ] Metricas adecuadas al problema (F1-Macro para multiclase desbalanceado)
- [ ] Analisis de errores por clase
- [ ] Feature importance interpretada
- [ ] Comparacion con modelo baseline

### Produccion
- [ ] App no entrena en produccion
- [ ] Manejo de errores para archivos faltantes
- [ ] Cache eficiente
- [ ] Datos privados excluidos del repositorio

### Documentacion
- [ ] Metodologia explicada (que y por que)
- [ ] Limitaciones reconocidas
- [ ] Instrucciones de reproduccion claras
- [ ] Metricas y figuras guardadas
