# Modelo Predictivo de Trayectorias Academicas

**XGBoost con tres modos de inferencia para prediccion del estado academico siguiente**
Universidad Tecnologica de Bolivar — Cartagena, Colombia

---

## Descripcion

Sistema predictivo basado en XGBoost que anticipa la transicion del estado academico de estudiantes de pregrado. Tres modos de inferencia:

- **Simulacion What-If (Modelo Automata)** (~99.99% F1-Macro): explorador de escenarios basado en el automata finito. Recibe OHE de estado + transicion como features — replica deterministicamente el automata (~100%). Ideal para simulacion de escenarios.
- **Prediccion Hibrida (Modelo Hibrido)** (~83.70% F1-Macro): combina indicadores numericos (PPA, creditos, cursos) con el estado academico actual del estudiante como feature categorica (OHE de 13 estados). Reduce parcialmente el data leakage del automata.
- **Prediccion real (Modelo Numerico)** (~52.17% F1-Macro): predictivo solo con datos academicos (PPA, creditos, cursos, programa). Sin conocimiento del automata. Resultados realistas.

Incluye **guardrails** (reglas de negocio) post-prediccion: filtro de sancion por PPA >= 3.0 anula estados de sancion (PAP, PAT, PFU, Recuperacion), y override automatico a Grado cuando CURSOS > 55 y CREDITOS > 150.

## Requisitos

- Python 3.10
- Dependencias: `pip install -r requirements.txt`

## Uso Rapido

```bash
# 1. Entrenar los tres modelos (requiere dataset local)
py -3.10 entrenar.py

# 2. Ejecutar la app
py -3.10 -m streamlit run app.py
```

## Estructura

```
├── app.py                  # Streamlit app (carga modelos, no entrena)
├── entrenar.py             # Entrenamiento de los 3 modelos
├── requirements.txt        # Dependencias
├── modelo/                 # Modelos .pkl (entrenados, para deploy)
│   ├── modelo_xgb.pkl / metadata.pkl              # Automata
│   ├── modelo_xgb_numerico.pkl / metadata_numerico.pkl  # Numerico
│   └── modelo_xgb_hibrido.pkl / metadata_hibrido.pkl    # Hibrido
├── outputs/                # Metricas, figuras, reportes
│   ├── (automata metrics)
│   ├── numerico/
│   └── hibrido/
└── docs/                   # Documentacion cientifica y guias
```

## App Streamlit

Tres pestanas:
1. **Prediccion Individual** — tres modos:
   - *Prediccion real*: solo datos academicos (sin estado)
   - *Prediccion real + Estado*: activa checkbox "Incluir Estado Actual como Feature Categorica" para modo Hibrido
   - *Simulacion What-If*: seleccionas estado + transicion (o todas las posibles) y ves escenarios
2. **Prediccion Masiva** — subir CSV/Excel. La app normaliza automaticamente nombres de columna (mayusculas, espacios, sinonimos como `ppa`/`promedio`, `materias`/`cursos`, etc.)
3. **Metricas** — comparacion lado a lado de los tres modelos, gap analysis, metricas por fold, importance, confusion matrix

### Guardrails aplicados
- PPA >= 3.0 → se anulan automaticamente estados de sancion (PAP, PAT, PFU, Recuperacion)
- CURSOS > 55 y CREDITOS > 150 → resultado forzado a "Grado"

## Papel Cientifico

La documentacion para el articulo academico esta en `docs/articulo_cientifico_estructura.md`, incluyendo metodologia detallada, resultados, discusion sobre data leakage, guardrails, y referencias.

## Datos

Los datasets contienen informacion privada de la universidad y no se distribuyen en este repositorio. La app utiliza valores por defecto para los dropdowns cuando el archivo de datos no esta presente.
