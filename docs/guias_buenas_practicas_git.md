# Guia de Buenas Practicas Git para Proyectos ML

## 1. Estructura del Repositorio

```
modeloPredictivoDelfinFinal/
├── app.py                  # Aplicacion Streamlit (NO entrenar aqui)
├── entrenar.py             # Script de entrenamiento (ejecutar una vez)
├── requirements.txt        # Dependencias para deploy
├── .gitignore              # Exclusiones (datos privados, modelos, cache)
│
├── modelo/                 # Modelos entrenados (necesarios para deploy)
│   ├── modelo_xgb.pkl           # Automata
│   ├── metadata.pkl             # Metadata Automata
│   ├── modelo_xgb_numerico.pkl  # Numerico
│   └── metadata_numerico.pkl    # Metadata Numerico
│
├── outputs/                # Metricas y figuras (versionadas en git)
│   ├── metrics.json
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── classification_report.json
│   └── numerico/
│       ├── metrics.json
│       ├── feature_importance.png
│       ├── confusion_matrix.png
│       └── classification_report.json
│
├── docs/                   # Documentacion
│   ├── articulo_cientifico_estructura.md
│   ├── guias_buenas_practicas_git.md
│   └── guia_ml_practicas.md
│
├── datos/                  # Datasets (IGNORADOS por git)
│   └── *.xlsx              # Datos privados de la universidad
│
└── .venv/                  # Entorno virtual (IGNORADO)
```

## 2. .gitignore Estandar para Proyectos ML

```
# Datasets (privados, no compartir)
datos/
data/
*.csv
*.xlsx

# Entorno virtual
.venv/
venv/
env/

# Cache de Python
__pycache__/
*.pyc
*.pyo
*.pyd

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
Thumbs.db
.DS_Store

# Modelos efimeros (si se regeneran frecuentemente)
# modelo/   # <-- NO ignorar si son necesarios para deploy
tmp_modelos/

# Logs y artefactos temporales
logs/
*.log
*.tmp
```

## 3. Convencion de Commits

Usar commits semanticos (Conventional Commits):

| Tipo | Uso | Ejemplo |
|---|---|---|
| `feat` | Nueva funcionalidad | `feat: add hybrid model training (automata + hybrid + numeric)` |
| `fix` | Correccion de bug | `fix: map integer predictions to class names` |
| `chore` | Mantenimiento | `chore: remove private datasets from repo` |
| `docs` | Documentacion | `docs: add scientific article structure guide` |
| `refactor` | Refactorizacion | `refactor: extract preprocessing to shared function` |
| `perf` | Optimizacion | `perf: reduce memory usage in batch prediction` |
| `test` | Tests | `test: add cross-validation group leakage assertion` |
| `style` | Formato | `style: format code with black` |

Reglas:
- Usar presente de indicativo ("add" no "added")
- Primera linea: max 72 caracteres
- Cuerpo opcional: explicar el QUE y el PORQUE, no el COMO
- No usar emojis

## 4. Flujo de Trabajo Recomendado

### 4.1 Desarrollo Local

```bash
# 1. Asegurar que estas en main
git checkout main
git pull

# 2. Crear rama de feature
git checkout -b feat/nueva-funcionalidad

# 3. Trabajar y commitear frecuentemente
git add archivo_modificado.py
git commit -m "feat: descripcion corta"

# 4. Mantener la rama actualizada con main
git fetch origin
git rebase origin/main

# 5. Push y crear PR
git push -u origin feat/nueva-funcionalidad
# Crear Pull Request en GitHub

# 6. Fusionar solo tras revision
# En GitHub: merge con squash para mantener historia limpia
```

### 4.2 Politica de Ramas

```
main              # Produccion (protegida: requiere PR + aprobacion)
  ├── dev         # Integracion (opcional para equipos grandes)
  ├── feat/*      # Nuevas funcionalidades
  ├── fix/*       # Correcciones
  └── docs/*      # Documentacion
```

## 5. Buenas Practicas Especificas para ML

### 5.1 Modelos Versionados

Los modelos .pkl entrenados deben versionarse en git si:
- Son necesarios para el deploy (Streamlit Cloud, Heroku, etc.)
- El script de entrenamiento es lento o requiere datos privados
- Se quiere reproducibilidad exacta de predicciones

NO versionar si:
- Se regeneran en CI/CD
- Ocupan > 100 MB (limite de GitHub)
- Contienen datos sensibles (usar `pickle` no es seguro)

### 5.2 Datos

Los datos NO se versionan en git. Alternativas:
- **DVC (Data Version Control):** Para datasets grandes con versionado
- **Almacenamiento externo:** S3, GCS, o servidor institucional
- **Script de descarga:** `scripts/download_data.py` con acceso controlado
- **Muestras sinteticas:** `datos/muestra_anonimizada.csv` para tests (si es posible)

### 5.3 Archivos de Configuracion

Evitar parametros hardcodeados. Usar archivos de configuracion (YAML/JSON):

```yaml
# config.yaml
model:
  name: xgboost_triple
  params:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 6

data:
  path: datos/08_solo_pregrado_automata_corregido_validado_v2.xlsx
  umbral_cursos: 55
  umbral_creditos: 150

cv:
  n_splits: 5
  strategy: StratifiedGroupKFold
```

### 5.4 Reproducibilidad

Fijar semillas aleatorias (random_state) en todos los componentes:

```python
random_state=42  # En XGBoost, StratifiedGroupKFold, SMOTE, etc.
```

Versionar `requirements.txt` con versiones exactas (opcional: `pip freeze > requirements.txt` tras pruebas).

## 6. Deploy en Streamlit Cloud

Configuracion del repositorio para deploy automatico:

```
Streamlit Cloud:
  - Repositorio: github.com/torres-palmerin/modeloPredictivoDelfinFinal
  - Branch: main
  - Archivo: app.py
  - Python version: 3.10 (especificar en streamlit config si es necesario)
```

Archivos REQUERIDOS en el repositorio para el deploy:
- [x] `app.py` — aplicacion principal
- [x] `requirements.txt` — dependencias
- [x] `modelo/modelo_xgb.pkl` — modelo entrenado
- [x] `modelo/metadata.pkl` — metadata del modelo
- [x] `modelo/modelo_xgb_numerico.pkl` — modelo alternativo
- [x] `modelo/metadata_numerico.pkl` — metadata alternativa
- [ ] `datos/*` — NO necesario (app usa valores por defecto)

## 7. Checklist de Buenas Practicas

- [ ] `.gitignore` configurado antes del primer commit
- [ ] Datos privados excluidos del repositorio
- [ ] Modelos incluidos (o regenerables via script)
- [ ] Commits semanticos y descriptivos
- [ ] Sin secretos/tokens en el codigo
- [ ] `requirements.txt` actualizado
- [ ] README o documentacion basica presente
- [ ] Semillas aleatorias fijadas
- [ ] Validacion cruzada sin leakage entre grupos
- [ ] La app no entrena en produccion
