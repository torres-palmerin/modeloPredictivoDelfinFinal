"""
==============================================================
SETUP AUTOMÁTICO — Modelo Predictivo de Trayectorias Académicas
==============================================================
Ejecuta este script UNA VEZ para instalar todo lo necesario.
Uso:
    python setup_proyecto.py
==============================================================
"""

import subprocess
import sys
import os

def run(cmd, desc):
    print(f"\n  → {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ Advertencia: {result.stderr.strip()[:200]}")
    else:
        print(f"  ✓ Listo")
    return result.returncode == 0

print("=" * 60)
print("SETUP — Modelo Predictivo de Trayectorias Académicas")
print("=" * 60)

# 1. Verificar Python
print(f"\n[1] Python detectado: {sys.version}")
if sys.version_info < (3, 8):
    print("  ✗ ERROR: Necesitas Python 3.8 o superior.")
    sys.exit(1)

# 2. Actualizar pip
run(f"{sys.executable} -m pip install --upgrade pip", "Actualizando pip")

# 3. Instalar dependencias
packages = [
    "pandas>=1.5",
    "numpy>=1.23",
    "scikit-learn>=1.2",
    "matplotlib>=3.6",
    "seaborn>=0.12",
    "xgboost>=1.7",
    "imbalanced-learn>=0.10",
    "openpyxl>=3.0",
    "nbformat>=5.7",
    "jupyterlab>=3.6",
    "ipykernel>=6.0",
]

print("\n[2] Instalando paquetes...")
for pkg in packages:
    name = pkg.split(">=")[0].split("==")[0]
    run(f"{sys.executable} -m pip install \"{pkg}\" -q", f"Instalando {name}")

# 4. Verificar imports
print("\n[3] Verificando instalación...")
checks = [
    ("pandas", "pd"),
    ("numpy", "np"),
    ("sklearn", "sklearn"),
    ("matplotlib", "plt"),
    ("seaborn", "sns"),
    ("xgboost", "xgb"),
    ("openpyxl", "openpyxl"),
    ("jupyterlab", "jupyterlab"),
]
all_ok = True
for mod, alias in checks:
    try:
        __import__(mod)
        print(f"  ✓ {mod}")
    except ImportError:
        print(f"  ✗ {mod} — NO instalado")
        all_ok = False

# 5. Crear estructura de carpetas
print("\n[4] Creando estructura de carpetas...")
carpetas = ["datos", "outputs", "notebooks"]
for c in carpetas:
    os.makedirs(c, exist_ok=True)
    print(f"  ✓ /{c}")

print("\n" + "=" * 60)
if all_ok:
    print("✓ Setup completado exitosamente.")
    print()
    print("PRÓXIMOS PASOS:")
    print("  1. Copia los archivos Excel a la carpeta  ./datos/")
    print("  2. Copia el notebook .ipynb a la carpeta  ./notebooks/")
    print("  3. Ejecuta JupyterLab con:")
    print()
    print("       jupyter lab")
    print()
    print("  4. Abre el notebook desde la carpeta notebooks/")
    print("  5. Ajusta las rutas de los archivos dentro del notebook")
    print("     (ver celda 'Rutas de archivos' al inicio)")
else:
    print("⚠ Algunos paquetes no se instalaron. Revisa los errores arriba.")
print("=" * 60)
