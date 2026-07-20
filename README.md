# Modelo Predictivo Dual de Trayectorias Academicas

**XGBoost + Automata Finito para prediccion del estado academico siguiente**
Universidad Tecnologica de Bolivar — Cartagena, Colombia

---

## Descripcion

Sistema predictivo dual basado en XGBoost que anticipa la transicion del estado academico de estudiantes de pregrado. Implementa dos modelos:

- **Modelo Automata** (~100% F1-Macro): incluye OHE del estado y transicion del automata como features. Sirve como referencia y diagnostico de data leakage estructural.
- **Modelo Numerico** (~52% F1-Macro): predice solo con datos academicos (PPA, creditos, cursos). Representa la capacidad predictiva realista.

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
1. **Prediccion Individual** — formulario con selector de modelo
2. **Prediccion Masiva** — subir CSV/Excel con prediccion batch
3. **Metricas** — comparacion lado a lado de ambos modelos

## Papel Cientifico

La documentacion para el articulo academico esta en `docs/articulo_cientifico_estructura.md`, incluyendo metodologia detallada, resultados, discusion sobre data leakage, y referencias.

## Datos

Los datasets contienen informacion privada de la universidad y no se distribuyen en este repositorio. La app utiliza valores por defecto para los dropdowns cuando el archivo de datos no esta presente.
