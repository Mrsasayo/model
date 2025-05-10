# Análisis y Modelo Predictivo de Condición de Productos

## Contexto

Este proyecto se basa en un conjunto de datos de una empresa de comercio electrónico. El dataset contiene información variada sobre los productos disponibles en la tienda.

## Ejercicio

El objetivo del ejercicio se divide en dos partes principales:
1.  **Análisis Exploratorio de Datos (EDA) y Limpieza:** Realizar las transformaciones necesarias para obtener un dataset limpio con las características requeridas para entrenar un modelo de aprendizaje automático.
2.  **Entrenamiento y Evaluación del Modelo:** Entrenar un modelo de aprendizaje automático que prediga si un producto es nuevo o usado, y extraer métricas para evaluar el rendimiento del modelo.

Se deben tomar todas las consideraciones y supuestos necesarios, realizar un EDA completo, utilizar figuras de apoyo y ejecutar todas las transformaciones necesarias. Para obtener el mejor modelo, se entrenarán diferentes clasificadores y se evaluará su rendimiento para seleccionar el mejor con base en las métricas extraídas.

## Fuente de Datos

El dataset original es `MLA_100k.jsonlines`, que contiene 100,000 registros de productos en formato JSON Lines.

## Estructura del Repositorio

El repositorio debería contener:
*   `eda_and_clean.ipynb`: Jupyter Notebook con el EDA y las transformaciones para obtener el dataset limpio.
*   `model_training.ipynb`: Jupyter Notebook con el pipeline de entrenamiento y evaluación de modelos.
*   `X_processed_dataset_26features.csv`: Dataset de características procesado y limpio, utilizado como entrada para el pipeline de modelado.
*   `y_target_dataset.csv`: Dataset con la variable objetivo (`condition_encoded`).
*   `README.md`: Este archivo, con las conclusiones y métricas del modelo.

## Análisis Exploratorio de Datos (EDA) y Limpieza de Datos (`eda_and_clean.ipynb`)

El proceso de EDA y limpieza se centró en transformar el dataset JSON crudo en un formato tabular estructurado y limpio, adecuado para el modelado.

1.  **Carga y Aplanamiento de Datos:**
    *   Se cargó el dataset `MLA_100k.jsonlines`.
    *   Se aplanaron las estructuras JSON anidadas (ej. `seller_address`, `shipping`, `attributes`) en columnas individuales. Esto resultó inicialmente en un DataFrame con 68 columnas.
    *   Columnas que contenían listas (ej. `sub_status`, `deal_ids`, `tags`, `coverage_areas`, `descriptions`, `pictures`) se convirtieron en representaciones de cadena o se procesaron para extraer información relevante (ej. `attributes_name`, `attributes_value_name`).
    *   Las columnas de fecha (`stop_time`, `start_time`, `last_updated`, `date_created`) se convirtieron a formato datetime donde fue apropiado para el EDA, aunque `date_created` y `last_updated` se mantuvieron como object para el preprocesamiento del modelo.

2.  **Análisis de Valores Nulos y Duplicados:**
    *   Se identificaron columnas con un alto porcentaje de valores nulos (ej. `coverage_areas` y `subtitle` con 100% nulos, `catalog_product_id` con ~99.9% nulos) y se eliminaron.
    *   Otras columnas con muchos nulos pero potencialmente útiles para imputación o como categoría "ausente" se mantuvieron inicialmente (ej. `warranty`, `video_id`).
    *   No se realizó una búsqueda explícita de duplicados en el notebook, pero la limpieza de IDs y la estructura de los datos sugieren que cada producto tiene un `id` único.

3.  **Análisis de Variables:**
    *   **Variable Objetivo (`condition`):** Se analizó su distribución, mostrando un ~53.7% de productos "new" y ~46.2% "used". Se codificó a `condition_encoded` (new=1, used=0).
    *   **Variables Numéricas:** Se analizaron descriptivamente `price`, `base_price`, `initial_quantity`, `sold_quantity`, `available_quantity`. `price` y `base_price` mostraron una alta asimetría, sugiriendo una transformación logarítmica para el modelado.
    *   **Variables Categóricas:** Se exploraron las categorías más frecuentes para `country`, `state`, `city`, `listing_type_id`, `buying_mode`, `shipping_mode`, `status`, `listing_source`, `international_delivery_mode`, `id_category`, `seller_id`.
    *   **Variables Booleanas:** Se analizó la distribución de `shipping_local_pick_up`, `free_shipping`, `accepts_mercadopago`, `automatic_relist`.
    *   **Variables de Texto/JSON:** Se observó la frecuencia de valores en `warranty` y `tags`.

4.  **Reducción de Dimensionalidad y Selección de Características (Inicial):**
    *   Se eliminaron columnas con alta cardinalidad o consideradas irrelevantes para predecir la condición del producto (IDs, URLs, comentarios, fechas no procesadas, etc.).
    *   El objetivo era obtener un conjunto de características más manejable y relevante. Se redujeron las 68 columnas iniciales a 26 características (+ la variable objetivo) que se guardaron en `X_processed_dataset_26features.csv`.
        *   Las columnas retenidas fueron: `price`, `base_price`, `initial_quantity`, `sold_quantity`, `available_quantity`, `listing_type_id`, `buying_mode`, `shipping_mode`, `status`, `listing_source`, `international_delivery_mode`, `country`, `state`, `city`, `id_category`, `seller_id`, `shipping_local_pick_up`, `free_shipping`, `accepts_mercadopago`, `automatic_relist`, `title`, `warranty`, `tags`, `shipping_methods`, `non_mercado_pago_payment_methods`, `date_created`.

5.  **Imputación y Transformaciones Específicas:**
    *   Para las columnas de texto que se mantuvieron (`warranty`, `tags`), los valores nulos se reemplazaron con placeholders como "Sin\_Garantia\_o\_Nulo\_Final" o "Sin\_Tags\_o\_Nulo\_Final".
    *   Las columnas numéricas (`price`, `base_price`, etc.) fueron estandarizadas y `price`/`base_price` transformadas logarítmicamente durante la limpieza final antes de la exportación de `X_processed_dataset_26features.csv`.
    *   Las columnas booleanas se convirtieron a enteros (0 o 1).

## Ingeniería de Características y Preprocesamiento para Modelado (`model_training.ipynb`)

A partir de `X_processed_dataset_26features.csv` y `y_target_dataset.csv`:

1.  **Identificación de Tipos de Características:**
    *   **Texto:** `title` (para TF-IDF).
    *   **Categóricas (Objetos no Texto):** `warranty`, `tags`, `shipping_methods`, `non_mercado_pago_payment_methods`, `date_created` (para OneHotEncoding).
    *   **Numéricas y Booleanas:** El resto de las 75 columnas (incluyendo las que ya eran numéricas/booleanas del CSV y las que se generan por OHE en el paso anterior de limpieza) se trataron como numéricas/booleanas. El notebook las vuelve a identificar y crea las booleanas a partir de las categóricas one-hot-encoded.

2.  **ColumnTransformer:** Se utilizó para aplicar diferentes transformaciones a distintos subconjuntos de columnas.
    *   **Procesamiento de Texto (`title`):**
        *   Imputación de nulos con un placeholder.
        *   `TfidfVectorizer` con `max_features=100`.
    *   **Codificación de Categóricas (objetos restantes):**
        *   Imputación de nulos con 'missing'.
        *   `OneHotEncoder` con `handle_unknown='ignore'` y `max_categories=10` (para limitar la dimensionalidad).
    *   **Características Numéricas:**
        *   Imputación de nulos con la mediana.
        *   Para el MLPClassifier, se aplicó adicionalmente `StandardScaler`.

    Este preprocesamiento resultó en un DataFrame con 215 características.

3.  **División de Datos:**
    *   Se dividieron los datos en conjuntos de entrenamiento (80%) y prueba (20%) usando `train_test_split` con `random_state=42` y estratificación por la variable objetivo.

## Entrenamiento y Evaluación de Modelos

Se entrenaron y evaluaron los siguientes modelos de clasificación:

1.  **Regresión Logística**
2.  **Random Forest Classifier**
3.  **Gradient Boosting Classifier**
4.  **MLP Classifier** (utilizando los datos numéricos escalados)

### Métricas de los Modelos

| Modelo                     | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :------------------------- | :------- | :-------------- | :----------- | :------------- | :----------------- | :-------------- | :---------------- |
| Regresión Logística        | 0.8451   | 0.8458          | 0.8423       | 0.8435         | 0.8454             | 0.8451          | 0.8447            |
| RandomForestClassifier     | 0.8784   | 0.8776          | 0.8782       | 0.8779         | 0.8786             | 0.8784          | 0.8785            |
| GradientBoostingClassifier | 0.8571   | 0.8563          | 0.8562       | 0.8562         | 0.8571             | 0.8571          | 0.8571            |
| MLPClassifier              | 0.8657   | 0.8649          | 0.8650       | 0.8649         | 0.8657             | 0.8657          | 0.8657            |

### Selección del Mejor Modelo

Basándose en las métricas de Accuracy y F1-Score (tanto Macro como Ponderado), el **RandomForestClassifier** fue el modelo con mejor rendimiento general, alcanzando una precisión (Accuracy) de aproximadamente 87.84% y un F1-Score Macro de 87.79%.

## Conclusiones

1.  El **Análisis Exploratorio de Datos (EDA)** fue fundamental para comprender la estructura, la calidad y las particularidades del dataset JSON anidado. Permitió identificar columnas con alta cardinalidad, muchos valores nulos o irrelevantes para la predicción.
2.  La **limpieza y preselección de características** realizada en `eda_and_clean.ipynb` fue un paso crucial para reducir la complejidad y preparar un conjunto de datos base (`X_processed_dataset_26features.csv`) más manejable y enfocado.
3.  La **ingeniería de características y preprocesamiento** adicional en `model_training.ipynb` (TF-IDF para títulos, OneHotEncoding para categóricas) expandió el espacio de características a 215, lo cual fue manejado eficientemente por los modelos.
4.  El **RandomForestClassifier** demostró ser el modelo más robusto y preciso para esta tarea de clasificación, superando a la Regresión Logística, Gradient Boosting y MLP.
5.  El modelo final puede predecir si un producto es "nuevo" o "usado" con una precisión general cercana al 88%, lo cual es un buen punto de partida para este tipo de problema.
6.  Las matrices de confusión (no mostradas aquí, pero generadas en el notebook) ayudarían a entender mejor los tipos de errores que comete cada modelo (falsos positivos vs. falsos negativos para cada clase).

## Posibles Mejoras Futuras

*   **Ajuste Fino de Hiperparámetros:** Utilizar técnicas como GridSearchCV o RandomizedSearchCV para optimizar los hiperparámetros del RandomForestClassifier o del MLPClassifier.
*   **Procesamiento de Texto Avanzado:** Explorar el uso de embeddings (Word2Vec, GloVe, FastText, o Transformers como BERT) para la columna `title` en lugar de TF-IDF, lo cual podría capturar mejor el significado semántico.
*   **Manejo de Fechas:** La columna `date_created` se trató como categórica después de la imputación. Se podría extraer información más granular (año, mes, día de la semana, antigüedad) como características numéricas.
*   **Análisis de Características Textuales Adicionales:** Considerar el preprocesamiento y uso de `warranty` y `tags` como características de texto si se sospecha que contienen información valiosa más allá de su simple presencia o categoría.
*   **Técnicas de Selección de Características Avanzadas:** Después de la codificación, aplicar técnicas como Recursive Feature Elimination (RFE) o selección basada en la importancia de características de modelos de árbol para reducir aún más la dimensionalidad si fuera necesario.
*   **Exploración de Otros Algoritmos:** Probar otros algoritmos como XGBoost, LightGBM, o modelos de Deep Learning más complejos si el rendimiento actual no es suficiente.
*   **Validación Cruzada:** Implementar validación cruzada durante el entrenamiento para obtener una estimación más robusta del rendimiento del modelo y para el ajuste de hiperparámetros.
