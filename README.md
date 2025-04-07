## prueba-analitica-modelo-opciones-de-pago
 Prueba Analítica: Predicción de gastos por categoría MCC

## Descripción 
Para esta prueba, se debe generar un modelo que ayude a predecir los gastos por categoría de los clientes. 
Para solucionarlo, se crearon 2 modelos de regresión con RandomForest usando la librería scikit-learn. En uno se usó el método de validación cruzada GridSearch y en el otro el método de validación cruzada RandomizedSearch. 

## Estructura del proyecto
.
├── data/
│   ├── transactions_data.csv            # Datos originales de transacciones
│   ├── users_data.csv                   # Datos originales de clientes
│   ├── transactions_data.parquet        # Versión optimizada de transacciones
│   └── users_data.parquet               # Versión optimizada de clientes
│
├── src/
│   └── modelo_regresion.py              # Script principal para procesamiento, entrenamiento y evaluación
│
├── requirements.txt                     # Dependencias del proyecto
├── README.md                            # Documentación del proyecto
└── .gitignore                           # Archivos y carpetas ignoradas por Git

## Requisitos o librerías
Incluye la versión 
joblib==1.4.2
numpy==1.24.4
pandas==2.0.3
python-dateutil==2.9.0.post0
pytz==2025.1
scikit-learn==1.3.2
scipy==1.10.1
six==1.17.0
threadpoolctl==3.5.0
tzdata==2025.1
statsmodels==0.13.5
pyarrow==10.0.1

## ¿Cómo ejecutar?
# Creacíon del archivo .gitignore
Se creó un archivo `.gitignore` para evitar que ciertos archivos y carpetas innecesarios se suban 
al repositorio. Esto incluye la carpeta `.venv/`, que contiene el ambiente virtual de Python 
y también la carpeta `data/`, que contiene los datos con los que se debe desarrollar el modelo.

# Creación del archivo requirements.txt 
Se creó un archivo `requirements.txt` que contiene todas las librerías que se usarán para desarrollar la prueba.

# Creación de carpeta data 
Se creó la carpeta data donde se pegaron los archivos descargados de Kaggle con los que se desarrollará la prueba.

# Creación y preparación del ambiente virtual en CMD
Se creó el ambiente virtual con: python -m venv .venv
Se activó el ambiente virtual con: .venv\Scripts\activate
Se actualizó el pip con: python -m pip install --upgrade pip
Se instalan todas las librerías que se usarán para desarrollar la prueba con: pip install -r requirements.txt

# Ejecutar el archivo principal

## Metodología 
Para construir un modelo que permita predecir los gastos por categoría de los clientes, se siguió el siguiente proceso:

1. Carga y transformación de datos
Se cargaron dos fuentes de datos: una de transacciones (transactions_data.csv) y otra de características de los usuarios (users_data.csv).

Ambos archivos se convirtieron al formato .parquet para mejorar la eficiencia en el manejo de datos.

Se realizó una unión entre las dos fuentes a través del identificador del cliente (client_id / id).

2. Limpieza de datos
Se eliminaron los símbolos monetarios ($) en variables de ingresos y montos.

Las columnas de ingresos y montos fueron convertidas a tipos numéricos (float o int).

3. Agregación por categoría
Se agruparon los datos por código de categoría (mcc) para generar una tabla agregada de características y montos por categoría.

Se calcularon estadísticas como el total gastado, número de clientes únicos, ingresos promedio, etc.

4. Definición de variables
Se definió amount como la variable objetivo (target) a predecir.

Las demás variables agregadas por categoría fueron utilizadas como variables predictoras.

5. División del dataset
Se dividió el dataset en conjunto de entrenamiento (70%) y prueba (30%) utilizando train_test_split.

6. Construcción del pipeline
Se creó un pipeline con los siguientes pasos:

Escalamiento: Normalización de variables numéricas con MinMaxScaler.

Interacciones: Generación de interacciones entre variables numéricas con PolynomialFeatures (solo interacciones, sin términos cuadráticos).

Selección de características: Eliminación de variables irrelevantes utilizando SelectFromModel con un modelo base RandomForestRegressor.

Modelo final: Entrenamiento de un RandomForestRegressor.

7. Optimización de hiperparámetros
Se utilizó GridSearchCV para buscar la mejor combinación de hiperparámetros (n_estimators, max_depth, etc.) con validación cruzada de 7 pliegues.

8. Evaluación del modelo
Se evaluó el rendimiento del modelo sobre el conjunto de prueba utilizando las siguientes métricas:

R² (Coeficiente de determinación): mide la proporción de la varianza explicada por el modelo.

RMSE (Raíz del Error Cuadrático Medio): da una idea del error promedio absoluto en unidades monetarias.

MAPE (Error Porcentual Absoluto Medio): expresa el error en porcentaje relativo al valor real.

## Resultados
Teniendo en cuenta el modelo de ML que obtuvo las mejores métricas (RFRegressor usando Gridsearch): 
Para esta prueba, seleccioné métricas que permiten evaluar tanto la capacidad explicativa del modelo como la magnitud del error en términos interpretables para el negocio.

Utilicé R² (Coeficiente de determinación), que indica qué proporción de la variabilidad del gasto por categoría es explicada por el modelo. En mi caso, obtuve un R² de 0.81, lo que muestra un buen poder predictivo.

También reporté el RMSE (Root Mean Squared Error), que representa el error promedio en las mismas unidades del gasto. Obtuve un RMSE de aproximadamente $4.204.221 por categoría, lo cual me permite dimensionar el margen de error esperado por predicción.

Como métrica adicional, calculé el MAPE (Mean Absolute Percentage Error), que mide el error relativo en porcentaje. En este caso fue alto (alrededor de 2.002 o también 200%), lo cual puede deberse a que algunas categorías tienen montos muy bajos, lo que distorsiona el cálculo del MAPE. Por esta razón, me enfoqué principalmente en R² y RMSE, que son más estables y representativos para este tipo de problema.

Considero que los resultados son sólidos y el modelo tiene un buen desempeño para predecir gastos por categoría de cliente.


