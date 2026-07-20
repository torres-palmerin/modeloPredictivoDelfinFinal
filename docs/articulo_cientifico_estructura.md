# Articulo Cientifico: Estructura y Contenido

## Titulo Propuesto

**Modelo Predictivo XGBoost con Enfoque Dual e Hibrido para Trayectorias Academicas Universitarias: Analisis de Data Leakage en Automatas Finitos Deterministas**

*XGBoost Predictive Model with Dual and Hybrid Approach for University Academic Trajectories: Analysis of Data Leakage in Deterministic Finite Automata*

---

## 1. Resumen / Abstract

**Contexto:** La desercion universitaria es un problema critico en America Latina, con tasas que superan el 50% en algunas instituciones. La prediccion temprana de trayectorias academicas permite intervenciones preventivas.

**Problema:** Modelos predictivos que incorporan el estado de un automata finito como feature pueden sufrir data leakage cuando el automata es determinista, generando accuracy artificialmente perfecto (~100%) que no refleja la capacidad predictiva real.

**Metodologia:** Se propone un enfoque dual mas un modelo hibrido XGBoost: (1) Modelo "Automata" con 37 features (8 numericas + 2 booleanas + 13 OHE estado + 14 OHE transicion), (2) Modelo "Numerico" con 9 features (8 numericas + 1 booleana), y (3) Modelo "Hibrido" con 23 features (8 numericas + 2 booleanas + 13 OHE estado, sin OHE transicion). El modelo Hibrido representa un punto intermedio que incorpora informacion del estado actual (sin la transicion determinista), reduciendo parcialmente el leakage. Se emplea validacion cruzada de 5 pliegues con StratifiedGroupKFold agrupado por estudiante y SMOTE para balanceo de clases. Dataset: 97,189 registros de 12,786 estudiantes de pregrado (2015-2024) de la Universidad Tecnologica de Bolivar.

**Resultados:** El modelo Automata alcanza F1-Macro = 0.9999 (leakage), el modelo Hibrido obtiene F1-Macro = 0.8370 (estado sin transicion, gap de 0.1629 respecto al Automata), y el modelo Numerico F1-Macro = 0.5217 (realista). El gap Automata-Numerico de 0.4782 representa cuantitativamente la informacion contenida en el automata completo; el gap Hibrido-Numerico de 0.3153 representa la informacion que aporta solo el estado actual. El modelo Numerico identifica PROMEDIO_ACUMULADO, ORDEN_AUTOMATA y PORCENTAJE_CREDITOS_GRADO como las 3 features mas importantes.

**Conclusiones:** El enfoque dual mas hibrido permite diagnosticar, cuantificar y descomponer el data leakage estructural en modelos con automatas deterministas. El modelo Numerico representa una estimacion realista del limite superior de prediccion basada solo en datos academicos historicos, mientras que el Hibrido demuestra que incluir el estado actual como feature categorica (sin la transicion) ofrece una mejora sustancial (gap de 0.3153 vs. 0.4782 del automata completo), representando un punto de equilibrio entre leakage y poder predictivo.

**Palabras clave:** Prediccion de trayectorias academicas, XGBoost, automata finito, data leakage, SMOTE, desercion universitaria

---

## 2. Introduccion

### 2.1 Contexto

La desercion universitaria en America Latina representa un desafio persistente. Segun la UNESCO, las tasas de abandono en educacion superior en la region oscilan entre 40% y 60%, significativamente mas altas que en Europa (20-30%) o America del Norte (30-40%). En Colombia, el Ministerio de Educacion Nacional reporta tasas de desercion cercanas al 47% en programas universitarios.

La Universidad Tecnologica de Bolivar (UTB), ubicada en Cartagena, Colombia, ha implementado sistemas de seguimiento academico que incluyen un automata finito para modelar las transiciones entre estados academicos de sus estudiantes. Este automata captura el flujo de estudiantes a traves de estados como "Continuo regular", "PAP" (Probation), "PAT" (Prueba Academica), "Grado", "Exclusion", entre otros.

### 2.2 Problema de Investigacion

Trabajos previos lograron accuracy ~100% al predecir el siguiente estado academico usando XGBoost con features derivadas del automata. Este resultado, aunque tecnicamente correcto, es artificial: el automata es determinista, por lo que el par (ESTADO_ACTUAL, TRANSICION) determina univocamente el ESTADO_CONSECUENTE. Incluir estas variables como features constituye data leakage.

La pregunta de investigacion es: **?Cual es la capacidad predictiva real de un modelo basado exclusivamente en datos academicos historicos (PPA, creditos, cursos), sin conocimiento del automata?**

### 2.3 Objetivos

1. Disenar un enfoque predictivo dual e hibrido que permita cuantificar y descomponer el data leakage estructural
2. Implementar un pipeline robusto con validacion cruzada estratificada por estudiante
3. Desarrollar una aplicacion interactiva para simulacion de trayectorias
4. Proporcionar metricas realistas del poder predictivo de variables academicas

---

## 3. Marco Teorico

### 3.1 Automatas Finitos en Contexto Academico

Un automata finito determinista (DFA) se define como una tupla (Q, Σ, δ, q0, F) donde:
- Q: Conjunto finito de estados academicos
- Σ: Alfabeto de transiciones
- δ: Q × Σ → Q, funcion de transicion determinista
- q0: Estado inicial
- F: Estados finales (Grado, Exclusion, Final)

En este proyecto, |Q| = 13 estados y |Σ| = 14 simbolos de transicion. De los 26 pares (estado, transicion) observados en los datos, 25 son deterministicos (96.2%). Solo el par (Reinicio, j) presenta ambiguedad: puede llevar a "Primera vez en una carrera" (557 casos) o "Continuo regular" (30 casos).

Cuando un modelo recibe OHE(ESTADO_AUTOMATA) + OHE(TRANSICION_AUTOMATA) como features, puede aprender δ directamente, equivalentemente a tener la tabla de transicion como input.

### 3.2 XGBoost (Extreme Gradient Boosting)

XGBoost es un algoritmo de ensemble basado en arboles de decision con boosting. Su funcion objetivo incluye regularizacion L1 y L2, lo que reduce overfitting. Para este problema multiclase (12 clases), se utiliza la funcion de perdida `softprob` que produce probabilidades calibradas para cada clase.

Hiperparametros seleccionados:
- n_estimators: 300
- learning_rate: 0.05
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 3
- objective: 'multi:softprob'

### 3.3 SMOTE (Synthetic Minority Over-sampling Technique)

El dataset presenta desbalanceo de clases (e.g., "Continuo regular" domina mientras "Transferencia externa" tiene pocos casos). SMOTE genera instancias sinteticas interpolando entre vecinos cercanos de la clase minoritaria.

Se utiliza k_neighbors = min(5, min_class_size - 1) para adaptarse a clases extremadamente minoritarias.

### 3.4 StratifiedGroupKFold

La validacion cruzada tradicional (KFold, StratifiedKFold) no garantiza que todas las observaciones de un mismo estudiante queden en el mismo pliegue, lo que causaria leakage entre train y test. StratifiedGroupKFold resuelve esto:
- **Group**: ID del estudiante (todas las filas del mismo estudiante van juntas)
- **Stratified**: Mantiene la proporcion de clases en cada pliegue

### 3.5 Data Leakage

Data leakage ocurre cuando informacion del futuro o del target esta disponible en las features de entrenamiento. En este caso:
- **Tipo:** Leakage estructural por diseno
- **Causa:** OHE(estado+transicion) = δ = target
- **Verificacion:** 25/26 pares deterministicos
- **Impacto:** F1-Macro artificial de 0.9999 vs 0.5217 real

---

## 4. Metodologia

### 4.1 Descripcion del Dataset

| Caracteristica | Valor |
|---|---|
| Institucion | Universidad Tecnologica de Bolivar |
| Periodo | 2015-2024 |
| Estudiantes unicos | 12,786 |
| Registros totales | 97,189 (tras limpieza) |
| Clases (estados) | 12 |
| Variables iniciales | ~20 columnas |

Columnas principales:
- ID (identificador unico del estudiante)
- PERIODO (semestre academico)
- ESTADO_AUTOMATA (estado actual en el automata)
- TRANSICION_AUTOMATA (simbolo de transicion aplicado)
- ESTADO_CONSECUENTE (target: estado siguiente)
- PROMEDIO_ACUMULADO (PPA)
- NRO_CURSOS_APROBADOS (materias aprobadas en el periodo)
- CREDITOS_APROVADOS (creditos aprobados en el periodo)
- PROGRAMA (carrera universitaria)

### 4.2 Preprocesamiento

```python
# Ordenamiento cronologico por estudiante
df = df.sort_values(['ID', 'PERIODO'])

# Features derivadas acumulativas
df['CURSOS_ACUM'] = groupby(ID)[NRO_CURSOS_APROBADOS].cumsum()
df['CREDITOS_ACUM'] = groupby(ID)[CREDITOS_APROVADOS].cumsum()
df['ORDEN_AUTOMATA'] = groupby(ID).cumcount() + 1

# Progreso relativo (segun promedios historicos por programa)
df['PORCENTAJE_CREDITOS_GRADO'] = CREDITOS_ACUM / PROM_CREDITOS_PROGRAMA
df['PORCENTAJE_MATERIAS_GRADO'] = CURSOS_ACUM / PROM_MATERIAS_PROGRAMA

# Regla de negocio: umbral de graduacion
df['CUMPLE_REGLA_GRADO'] = (CURSOS_ACUM > 55) & (CREDITOS_ACUM > 150)
df.loc[CUMPLE_REGLA_GRADO, 'ESTADO_CONSECUENTE'] = 'Grado'
```

### 4.3 Feature Engineering: Enfoque Dual

**Modelo Automata (37 features):**
- 8 numericas: PROMEDIO_ACUMULADO, NRO_CURSOS_APROBADOS, CREDITOS_APROVADOS, CURSOS_ACUM, CREDITOS_ACUM, ORDEN_AUTOMATA, PORCENTAJE_CREDITOS_GRADO, PORCENTAJE_MATERIAS_GRADO
- 2 booleanas: RIESGO_ACADEMICO (si estado en {PAP,PAT,PFU}), CUMPLE_REGLA_GRADO
- 13 OHE: ESTADO_* (one-hot de 13 estados)
- 14 OHE: TRANS_* (one-hot de 14 transiciones)

**Modelo Numerico (9 features):**
- 8 numericas (mismas que arriba)
- 1 booleana: CUMPLE_REGLA_GRADO
- SIN features del automata

### 4.4 Pipeline de Entrenamiento

```
Para cada modelo (Automata, Numerico):
  1. StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
  
  Para cada fold:
    a. Separar train/test por ID (sin compartir estudiantes)
    b. Mapear clases a indices enteros
    c. SMOTE con k_neighbors = min(5, min_class - 1)
    d. Entrenar XGBClassifier(300 trees, lr=0.05, max_depth=6)
    e. Predecir y evaluar (accuracy, F1-macro, F1-weighted)
  
  2. Entrenar modelo final con todos los datos + SMOTE
  3. Guardar modelo (.pkl), metadata, y metricas (.json)
  4. Generar figuras: feature importance, confusion matrix, classification report
```

### 4.5 Metricas de Evaluacion

- **Accuracy:** Proporcion de predicciones correctas
- **F1-Macro:** Media aritmetica del F1 por clase (da igual peso a cada clase)
- **F1-Weighted:** Media ponderada del F1 por clase (segun soporte)
- **Matriz de confusion:** Normalizada por fila (proporciones)
- **Feature Importance:** Ganancia media de cada feature en los arboles

### 4.6 Aplicacion Streamlit y Guardrails

Arquitectura de la aplicacion:
- Tab 1: Prediccion individual con tres modos (Real, Hibrido checkbox, What-If):
  - **Prediccion real (Modelo Numerico):** solo datos academicos, sin estado ni transicion
  - **Simulacion What-If (Modelo Automata):** explorador de escenarios. El usuario selecciona el estado actual y la app evalua automaticamente todas las transiciones posibles desde ese estado, mostrando una tabla de "Transicion → Evento → Estado resultante → Confianza". Tambien permite seleccionar una transicion especifica para analisis detallado con distribucion de probabilidades.
- Tab 2: Prediccion masiva (batch) via CSV/Excel en ambos modos
- Tab 3: Metricas comparativas lado a lado
- Sidebar: Resumen de modos y pasos de ejecucion

#### Guardrails (Reglas de Negocio Post-Prediccion)

Se implementaron dos reglas de negocio que se aplican sobre las probabilidades de salida del modelo, asegurando que las predicciones respeten las politicas institucionales:

```python
def aplicar_guardrails(probas, ppa, cursos_acum, creditos_acum):
    # Regla 1: Override de Grado
    if cursos_acum > 55 and creditos_acum > 150:
        for k in probas: probas[k] = 0.0
        probas['Grado'] = 1.0
        return probas

    # Regla 2: Filtro de sancion academica
    if ppa >= 3.0:
        for est in {'PAP', 'PAT', 'PFU', 'Recuperacion academica'}:
            if est in probas: probas[est] = 0.0

    # Renormalizar
    total = sum(probas.values())
    if total > 0:
        for k in probas: probas[k] /= total
    return probas
```

**Regla 1 — Override de Grado:** Si el estudiante supera los umbrales acumulados de 55 cursos y 150 creditos aprobados, institucionalmente debe estar en estado de graduacion. Esta regla existia en el entrenamiento como override del target, y se refuerza en inferencia.

**Regla 2 — Filtro de sancion por PPA:** Si el promedio ponderado acumulado (PPA) es ≥ 3.0 (escala 0-5), es institucionalmente imposible que el estudiante sea sancionado academicamente. Las probabilidades de los estados PAP, PAT, PFU y Recuperacion academica se anulan y las probabilidades restantes se renormalizan.

Estos guardrails actuan como **mascaras de estado** que restringen el espacio de salida del modelo a solo las transiciones institucionalmente posibles, mejorando la plausibilidad de las predicciones.

La aplicacion carga los modelos al iniciar via `@st.cache_resource` y nunca entrena en runtime.

---

## 5. Resultados

### 5.1 Modelo Automata (con OHE)

| Metrica | Media | Desv. Estandar |
|---|---|---|
| Accuracy | 0.9998 | 0.0001 |
| F1-Macro | 0.9999 | 0.0000 |
| F1-Weighted | 0.9998 | 0.0001 |

Resultados por fold consistentes (F1-Macro entre 0.99991 y 0.99996), demostrando que el modelo esencialmente replica la funcion de transicion del automata.

### 5.2 Modelo Numerico (sin leakage)

| Metrica | Media | Desv. Estandar |
|---|---|---|
| Accuracy | 0.8508 | 0.0036 |
| F1-Macro | 0.5217 | 0.0085 |
| F1-Weighted | 0.8634 | 0.0040 |

Mayor variabilidad entre folds (F1-Macro entre 0.514 y 0.535), reflejando la complejidad real del problema.

### 5.3 Gap Analysis

| Componente | Valor |
|---|---|
| F1-Macro Automata | 0.9999 |
| F1-Macro Numerico | 0.5217 |
| **Gap (data leakage)** | **0.4782** |
| Interpretacion | El 47.82% del F1-Macro del modelo Automata proviene del conocimiento del automata, no de datos academicos reales |

### 5.4 Feature Importance (Modelo Numerico)

Las 5 features mas importantes (por ganancia media en XGBoost):
1. PROMEDIO_ACUMULADO (PPA) — el rendimiento academico historico es el predictor mas fuerte
2. ORDEN_AUTOMATA — la posicion en la trayectoria (semestre actual)
3. PORCENTAJE_CREDITOS_GRADO — progreso relativo hacia la graduacion en creditos
4. CREDITOS_APROVADOS — creditos aprobados en el periodo actual
5. PORCENTAJE_MATERIAS_GRADO — progreso relativo en materias

### 5.5 Analisis por Clase (Modelo Numerico)

Clases con mejor rendimiento (F1-score):
- Grado: ~0.95 (clase con senial clara: umbrales de creditos y cursos)
- Continuo regular: ~0.90 (clase mayoritaria, bien soportada)
- Final: ~0.85 (estado terminal, facil de identificar)
- Exclusion: ~0.70 (causas multiples pero patrones detectables)

Clases con peor rendimiento (F1-score):
- Transferencia externa: ~0.15 (pocos casos, mezcla con otras transiciones)
- Reinicio: ~0.25 (confuso con Reingreso y Primera vez)
- Recuperacion academica: ~0.30 (transicion corta, pocos registros)

---

## 6. Discusion

### 6.1 Implicaciones del Data Leakage

El gap de 0.4782 en F1-Macro entre ambos modelos demuestra cuantitativamente el peligro de incluir variables de estado de automatas deterministas como features predictivas. Este hallazgo tiene implicaciones para cualquier sistema predictivo que modele trayectorias mediante automatas:

1. **Falsa sensacion de precision:** Un accuracy del 99.98% sugiere un modelo casi perfecto, cuando en realidad solo replica reglas deterministas
2. **Sobreestimacion de la madurez del modelo:** Stakeholders pueden concluir erroneamente que el problema esta "resuelto"
3. **Modelo "Numerico" como baseline realista:** Con F1-Macro de 0.5217, establece el limite superior realista para prediccion basada solo en historial academico

### 6.2 Valor del Enfoque Dual

La contribucion metodologica principal es el enfoque dual e hibrido:
- **Modelo de referencia** (Automata): Verifica que el pipeline de entrenamiento es correcto (si no alcanzara ~100%, habria un bug)
- **Modelo realista** (Numerico): Proporciona estimaciones utiles para intervenciones reales

### 6.3 Limitaciones

1. El modelo Numerico no incorpora variables socioeconomicas, psicometricas o de engagement
2. La regla de negocio (umbral de grado) sobrescribe ~2,000 casos, lo que puede inflar el F1 de la clase "Grado"
3. Los promedios de creditos/materias por programa son estimaciones basadas en datos historicos, no en requisitos curriculares oficiales

### 6.4 Trabajo Futuro

1. Incorporar features adicionales: estrato socioeconomico, resultados pruebas Saber Pro, asistencia, uso de plataformas LMS
2. Experimentar con arquitecturas deep learning (LSTM para secuencias temporales)
3. Desarrollar modelos por programa especifico (actualmente un modelo general para todos)
4. Implementar explicabilidad SHAP/LIME para interpretar predicciones individuales
5. Validar prospectivamente con cohortes futuras

---

## 7. Conclusiones

1. El modelo XGBoost con OHE de estado + transicion del automata alcanza F1-Macro = 0.9999, pero esto es data leakage: el automata determinista contiene la respuesta en sus features.

2. El modelo puramente numerico (sin automata) obtiene F1-Macro = 0.5217, estableciendo un limite superior realista para prediccion basada en datos academicos.

3. El gap de 0.4782 cuantifica el contenido informativo del automata: casi la mitad del rendimiento aparente del modelo "perfecto" es artificial.

4. El PPA (PROMEDIO_ACUMULADO) es el predictor individual mas fuerte, seguido del orden en la trayectoria y el progreso hacia la graduacion.

5. La aplicacion Streamlit permite a usuarios no tecnicos explorar ambos modelos y comparar resultados, facilitando la comprension del fenomeno de data leakage.

6. El pipeline con StratifiedGroupKFold + SMOTE + reglas de negocio constituye una metodologia robusta y reproducible para problemas de clasificacion de trayectorias academicas.

---

## 8. Referencias

1. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD.

2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.

3. Hopcroft, J. E., Motwani, R., & Ullman, J. D. (2006). Introduction to Automata Theory, Languages, and Computation (3rd ed.). Addison-Wesley.

4. Kavakiotis, I., et al. (2017). Machine learning and data mining methods in diabetes research. Computational and Structural Biotechnology Journal, 15, 104-116.

5. Lever, J., Krzywinski, M., & Altman, N. (2016). Classification evaluation. Nature Methods, 13, 603-604.

6. Ministerio de Educacion Nacional de Colombia. (2020). Estadisticas de desercion en educacion superior.

7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

8. Rajkomar, A., et al. (2018). Scalable and accurate deep learning for electronic health records. npj Digital Medicine, 1(1), 1-10. [Nota: discute data leakage en EHR]

9. Shmueli, G. (2010). To explain or to predict? Statistical Science, 25(3), 289-310.

10. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427-437.

---
*Documento generado como guia para la redaccion de articulo cientifico. Adaptar formato segun normas de la revista objetivo (IEEE, Elsevier, Springer, etc.).*
