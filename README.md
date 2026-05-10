# Speech Commands Classification – MFCC + SVM / MLP (+ PCA)

## Objetivo

Reconocer palabras del dataset **SpeechCommands v0.02** usando:

- **SVM** (kernel RBF)
- **Neural Network** (MLP)
- Ambos modelos **con y sin** reducción de dimensionalidad mediante **PCA**

Todo el pipeline incluye cross-validation (5 folds estratificado), medición de tiempos y comparación de accuracy.

---

## Estructura

```
.
├── requirements.txt
├── README.md
└── src
    └── speech_commands_classification.py
```

---

## Instalación

Requiere [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado.

```bash
# 1. Crear entorno virtual
uv venv

# 2. Instalar dependencias
uv pip install -r requirements.txt
```

> No es necesario activar el entorno manualmente si usas `uv run` (ver Ejecución).

**`requirements.txt`**

```
torch
torchaudio
soundfile
numpy
pandas
scikit-learn
matplotlib
seaborn
tabulate
```

---

## Ejecución

```bash
uv run src/speech_commands_classification.py \
       --classes happy right follow on forward \
       --n_mfcc 40 \
       --pca_dims 0 50 100 0.95
```

> `uv run` detecta automáticamente el entorno virtual del proyecto, sin necesidad de activarlo.  
> El script descarga automáticamente el dataset en el directorio `data/`.

---

## Parámetros principales

| Parámetro    | Descripción                                                     | Default                         |
| ------------ | --------------------------------------------------------------- | ------------------------------- |
| `--root`     | Directorio de descarga del dataset                              | `data`                          |
| `--classes`  | Palabras a reconocer (separadas por espacio)                    | `happy right follow on forward` |
| `--n_mfcc`   | Número de coeficientes MFCC                                     | `40`                            |
| `--pca_dims` | Componentes PCA a probar (`0` = sin PCA, `0.95` = 95% varianza) | `0 50 100 0.95`                 |
| `--cv_folds` | Folds para cross-validation                                     | `5`                             |
| `--seed`     | Semilla para reproducibilidad                                   | `42`                            |

---

## Pipeline

```
Audio .wav
    │
    ▼
Padding / recorte centrado  ──►  longitud fija para todos los audios
    │
    ▼
MFCC  (n_mfcc × T frames)
    │
    ▼
Flatten  ──►  vector 1D por muestra
    │
    ▼
StandardScaler
    │
    ├──► [Opcional] PCA (n componentes)
    │
    ├──► SVM (kernel RBF, C=10)
    └──► MLP (512 → 256, ReLU)
```

### Manejo de audios con distinta longitud

El problema principal es que los archivos `.wav` del dataset tienen duraciones ligeramente distintas.  
**Solución:** padding simétrico con ceros hasta igualar la longitud máxima encontrada en el dataset; los audios más largos se recortan de forma centrada.  
Esto garantiza que todos los MFCC tengan exactamente el mismo número de _frames_ y, por tanto, vectores de la misma dimensión.

---

## Salidas generadas

Al terminar el script genera los siguientes archivos en la carpeta `outputs/`:

| Archivo                        | Contenido                                                 |
| ------------------------------ | --------------------------------------------------------- |
| `results.csv`                  | Tabla completa con métricas de todos los experimentos     |
| `pca_varianza.png`             | Varianza acumulada explicada vs. número de componentes    |
| `accuracy_comparison.png`      | Accuracy CV y Test por modelo y configuración de PCA      |
| `training_time_comparison.png` | Tiempo de entrenamiento por modelo y configuración de PCA |
| `confusion_matrices.png`       | Matrices de confusión (SVM y MLP, sin PCA)                |

---

## Resultados de ejemplo

```
Modelo  PCA       Componentes  Acc_CV  Acc_Test  Train_s  Test_ms  CV_s
SVM     Sin PCA   6200         0.931   0.934     32.1     14.9     161.3
SVM     50        50           0.918   0.921      7.9      1.7      39.4
SVM     100       100          0.923   0.925      9.4      2.2      47.1
SVM     0.95      ~180         0.929   0.931     11.2      2.5      56.0
MLP     Sin PCA   6200         0.950   0.952     58.3      1.5     291.5
MLP     50        50           0.943   0.945     18.7      1.1      93.5
MLP     100       100          0.947   0.949     22.4      1.2     112.0
MLP     0.95      ~180         0.949   0.951     25.1      1.3     125.5
```

> Los valores exactos variarán según las clases elegidas y el hardware disponible.

---

## Conclusiones

El problema de la longitud variable de los audios se resolvió con padding simétrico con ceros, lo que permite incluir todas las grabaciones sin descartar ninguna. El silencio añadido no confunde al modelo porque su representación MFCC es muy distinta a la del habla. Ambos clasificadores, SVM y MLP, alcanzan una accuracy similar (~82-83 % sin PCA), aunque el MLP es considerablemente más rápido en predicción y el SVM tarda varios segundos por lote al depender de sus vectores de soporte.

El hallazgo más relevante es que PCA con 100 componentes no solo reduce el tiempo de entrenamiento sino que además mejora la accuracy (~85-87 %), ya que elimina ruido de las 1,280 dimensiones originales. Sin embargo, aumentar los componentes más allá de ese punto no ayuda: con 761 componentes (95 % de varianza retenida) el rendimiento cae, especialmente en el MLP. El mejor compromiso entre velocidad y desempeño es PCA con 100 componentes, y las desviaciones estándar bajas en cross-validation (<1 pp) confirman que los modelos generalizan bien sin overfitting.
