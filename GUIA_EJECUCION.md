# Guía para ejecutar el Modelo Predictivo de Trayectorias Académicas en tu PC

## Archivos que necesitas

Descarga estos archivos de la conversación con Claude:

| Archivo | Descripción |
|---------|-------------|
| `Modelo_Predictivo_Trayectorias.ipynb` | Notebook principal (Jupyter) |
| `pipeline_automata.py` | Script Python independiente |
| `setup_proyecto.py` | Instalador automático |
| `12_only_undergraduate_with_automaton.xlsx` | Dataset principal |
| `07_undergraduate_pathway_with_degree_automaton.xlsx` | Dataset graduados |

---

## Opción A — JupyterLab (recomendado para presentar)

### Paso 1: Instalar Python

Si aún no tienes Python instalado:

- **Windows / Mac:** Descarga desde https://www.python.org/downloads/
  - Versión mínima: **Python 3.8**
  - Windows: al instalar, marca la casilla **"Add Python to PATH"**
- **Si ya tienes Anaconda/Miniconda:** perfecto, ya tienes todo.

Verifica en la terminal:

```
python --version
```

Debe mostrar algo como `Python 3.10.x`

---

### Paso 2: Instalar las dependencias automáticamente

1. Abre una **terminal** (Windows: `cmd` o `PowerShell`; Mac: `Terminal`)
2. Navega a la carpeta donde descargaste los archivos:

```bash
cd C:\Users\TuNombre\Descargas\proyecto_trayectorias
```

3. Ejecuta el setup:

```bash
python setup_proyecto.py
```

Esto instala automáticamente:
- `pandas`, `numpy` — manejo de datos
- `scikit-learn` — modelos de ML
- `xgboost` — modelo ganador del proyecto
- `matplotlib`, `seaborn` — visualizaciones
- `openpyxl` — lectura de archivos Excel
- `jupyterlab` — entorno de notebooks

---

### Paso 3: Organizar los archivos

Crea esta estructura de carpetas:

```
proyecto_trayectorias/
│
├── datos/
│   ├── 12_only_undergraduate_with_automaton.xlsx
│   └── 07_undergraduate_pathway_with_degree_automaton.xlsx
│
├── notebooks/
│   └── Modelo_Predictivo_Trayectorias.ipynb
│
├── outputs/           ← aquí se guardarán las figuras generadas
│
└── setup_proyecto.py
```

---

### Paso 4: Ajustar las rutas en el notebook

Abre el notebook y en la **primera celda de código** cambia las rutas:

```python
# ANTES (rutas del servidor de Claude)
FILE_MAIN = '/mnt/user-data/uploads/12_only_undergraduate_with_automaton.xlsx'
FILE_GRAD = '/mnt/user-data/uploads/07_undergraduate_pathway_with_degree_automaton.xlsx'
OUTPUT_DIR = './outputs'

# DESPUÉS (rutas en tu PC)
FILE_MAIN = '../datos/12_only_undergraduate_with_automaton.xlsx'
FILE_GRAD = '../datos/07_undergraduate_pathway_with_degree_automaton.xlsx'
OUTPUT_DIR = '../outputs'
```

> Si el notebook está en la misma carpeta que los Excel, puedes usar simplemente el nombre del archivo:
> ```python
> FILE_MAIN = '12_only_undergraduate_with_automaton.xlsx'
> ```

---

### Paso 5: Lanzar JupyterLab

En la terminal, desde la carpeta del proyecto:

```bash
jupyter lab
```

Se abrirá automáticamente el navegador en `http://localhost:8888`.

1. Navega a `notebooks/`
2. Abre `Modelo_Predictivo_Trayectorias.ipynb`
3. Ejecuta todo con: **Kernel → Restart & Run All**

---

## Opción B — Script Python directo (sin Jupyter)

Si prefieres ejecutar desde la terminal sin abrir Jupyter:

```bash
python pipeline_automata.py
```

Las figuras y resultados se guardan en la carpeta `outputs/`.

---

## Opción C — Google Colab (sin instalar nada)

Si no quieres instalar nada localmente, usa Google Colab:

1. Ve a https://colab.research.google.com
2. Sube el archivo `Modelo_Predictivo_Trayectorias.ipynb`
3. Sube los dos archivos Excel (usando el panel lateral de Colab)
4. Cambia las rutas en el notebook a:

```python
FILE_MAIN = '/content/12_only_undergraduate_with_automaton.xlsx'
FILE_GRAD = '/content/07_undergraduate_pathway_with_degree_automaton.xlsx'
OUTPUT_DIR = '/content/outputs'
```

5. Clic en **Entorno de ejecución → Ejecutar todo**

> Los paquetes como `xgboost`, `sklearn`, `pandas` ya vienen preinstalados en Colab.

---

## Problemas comunes

### "python no se reconoce como comando" (Windows)

Reinstala Python desde python.org y marca **"Add Python to PATH"** durante la instalación. O usa `py` en lugar de `python`.

### Error al leer los archivos Excel

Asegúrate de que:
1. Los archivos están en la ruta que indicaste
2. Los nombres no tienen espacios extra
3. `openpyxl` está instalado: `pip install openpyxl`

### XGBoost no instala en Windows

```bash
pip install xgboost --pre
```

O usa el modelo alternativo — el notebook automáticamente usa `GradientBoosting` si XGBoost no está disponible.

### El notebook tarda mucho (dataset grande)

El dataset tiene 106K filas. La primera ejecución puede tardar entre **3 y 8 minutos** dependiendo del PC. Es normal.

---

## Tiempo estimado de ejecución

| Paso | Tiempo aproximado |
|------|-------------------|
| Carga de datos | 30–60 seg |
| Construcción del dataset de entrenamiento | 20–40 seg |
| Ingeniería de features | 5–10 seg |
| Entrenamiento de 4 modelos | 3–6 min |
| Generación de 8 figuras | 30–60 seg |
| **Total** | **~5–8 min** |

---

## Estructura de las salidas generadas

Al ejecutar, se crearán en `outputs/`:

```
outputs/
├── 01_comparacion_modelos.png       ← barras comparando los 4 modelos
├── 02_confusion_matrix.png          ← matriz de confusión (absoluta + normalizada)
├── 03_f1_por_clase.png              ← F1 por cada estado académico
├── 04_feature_importance.png        ← importancia de variables
├── 05_distribucion_target.png       ← distribución del target
├── 06_trayectorias_frecuentes.png   ← top 15 trayectorias
├── 07_automata_diagram.png          ← diagrama visual del autómata
├── 08_tabla_metricas.png            ← tabla resumen de métricas
├── metrics.json                     ← todas las métricas en formato JSON
├── training_dataset.csv             ← dataset de entrenamiento generado
└── trayectorias_por_estudiante.csv  ← trayectoria completa por alumno
```
