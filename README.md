# Modelo Predictivo Dual de Trayectorias Academicas

**XGBoost + Automata Finito para prediccion del estado academico siguiente**
Universidad Tecnologica de Bolivar — Cartagena, Colombia

---

## Descripcion

Sistema predictivo dual basado en XGBoost que anticipa la transicion del estado academico de estudiantes de pregrado. Dos modos de uso:

- **Prediccion real (Modelo Numerico)** (~52% F1-Macro): predictivo solo con datos academicos (PPA, creditos, cursos, programa). Sin conocimiento del automata. Resultados realistas.
- **Simulacion What-If (Modelo Automata)** (~100% F1-Macro): explorador de escenarios basado en el automata finito. Permite simular el efecto de diferentes transiciones sobre el estado academico.

Incluye **guardrails** (reglas de negocio) post-prediccion: filtro de sancion por PPA y override por umbrales de grado.

## Requisitos

- Python 3.10
- Dependencias: `pip install -r requirements.txt`

## Uso Rapido

```bash
# 1. Entrenar ambos modelos (requiere dataset local)
py -3.10 entrenar.py

# 2. Ejecutar la app
py -3.10 -m streamlit run app.py
```

## Estructura

```
├── app.py                  # Streamlit app (carga modelos, no entrena)
├── entrenar.py             # Entrenamiento dual (ejecutar una vez)
├── requirements.txt        # Dependencias
├── modelo/                 # Modelos .pkl (entrenados, para deploy)
├── outputs/                # Metricas, figuras, reportes
└── docs/                   # Documentacion cientifica y guias
```

## App Streamlit

Tres pestanas:
1. **Prediccion Individual** — dos modos:
   - *Prediccion real*: solo datos academicos, sin estado/transicion
   - *Simulacion What-If*: seleccionas estado + transicion (o todas las posibles) y ves escenarios
2. **Prediccion Masiva** — subir CSV/Excel con prediccion batch en ambos modos
3. **Metricas** — comparacion lado a lado de ambos modelos

### Guardrails aplicados
- PPA >= 3.0 → se anulan automaticamente estados de sancion (PAP, PAT, PFU, Recuperacion)
- CURSOS > 55 y CREDITOS > 150 → resultado forzado a "Grado"

## Papel Cientifico

La documentacion para el articulo academico esta en `docs/articulo_cientifico_estructura.md`, incluyendo metodologia detallada, resultados, discusion sobre data leakage, guardrails, y referencias.

## Datos

Los datasets contienen informacion privada de la universidad y no se distribuyen en este repositorio. La app utiliza valores por defecto para los dropdowns cuando el archivo de datos no esta presente.
