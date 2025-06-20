# --------------------------------------------------------------------------
# SCRIPT DE ENTRENAMIENTO PARA RANDOM FOREST (RF)
# --------------------------------------------------------------------------
# Objetivo: Entrenar y optimizar un modelo Random Forest para la clasificación
#           de regiones genómicas.
# Autor: Daniel González Palazón
# Fecha: 15/06/2025
# --------------------------------------------------------------------------


# --- 1. CARGA DE LIBRERÍAS ---
# Se cargan los paquetes necesarios para el proceso.
# - caret: Framework principal para el modelado de Machine Learning.
# - ranger: Implementación de Random Forest altamente eficiente, ideal para clúster.
# - doParallel: Necesario para ejecutar el entrenamiento en paralelo y reducir tiempos.
# --------------------------------------------------------------------------
library(caret)
library(ranger)
library(doParallel)


# --- 2. CONFIGURACIÓN DEL ENTORNO PARALELO ---
# Se configura la computación en paralelo para aprovechar los recursos del clúster.
# El script lee la variable de entorno 'SLURM_NTASKS', que el gestor de trabajos
# SLURM define con el número de CPUs asignadas al trabajo.
# Si el script se ejecuta localmente (fuera de SLURM), usará todos los cores
# disponibles menos uno como medida de seguridad.
# --------------------------------------------------------------------------
num_cores <- as.numeric(Sys.getenv("SLURM_NTASKS", unset = 1))
if (num_cores == 1) {
  # Fallback para ejecución local si la variable de SLURM no está definida
  num_cores <- detectCores() - 1
}
cl <- makeCluster(num_cores)
registerDoParallel(cl)
cat(paste("-> Registrados", getDoParWorkers(), "cores para el entrenamiento en paralelo.\n"))


# --- 3. CARGA DE DATOS ---
# Se carga el conjunto de entrenamiento, que fue previamente creado y guardado
# en el script de preparación de datos.
# --------------------------------------------------------------------------
cat("-> Cargando el conjunto de entrenamiento 'train_set.rds'...\n")
train_set <- readRDS("train_set.rds")


# --- 4. DEFINICIÓN DE LA ESTRATEGIA DE ENTRENAMIENTO ---
# Se define un objeto 'trainControl' que encapsula la metodología de validación
# y optimización del modelo.
# --------------------------------------------------------------------------
train_control <- trainControl(
  method = "repeatedcv", # Validación Cruzada Repetida: Proporciona una estimación del error más estable.
  number = 10,           # 10 folds: Un estándar que equilibra el sesgo y la varianza de la estimación.
  repeats = 3,           # 3 repeticiones: Se promedia el rendimiento sobre 30 particiones (10x3), aumentando la robustez.
  verboseIter = TRUE,    # Muestra un log del progreso del entrenamiento.
  allowParallel = TRUE,  # Habilita la paralelización configurada previamente.
  classProbs = TRUE,     # Necesario para calcular métricas de probabilidad como AUC.
  summaryFunction = multiClassSummary, # Devuelve un set completo de métricas para problemas multiclase.
  sampling = "down"      # Down-sampling: En cada una de las 30 iteraciones, se submuestrea la clase
  # mayoritaria para igualarla a la siguiente. Esto es crucial para forzar al
  # modelo a aprender de las clases minoritarias.
)
cat("-> Estrategia de entrenamiento definida: Validación cruzada repetida (10 folds, 3 repeticiones) con down-sampling.\n")


# --- 5. DEFINICIÓN DEL GRID DE HIPERPARÁMETROS ---
# Se crea una rejilla (grid) con los valores de los hiperparámetros a explorar.
# La selección de estos valores es una decisión de diseño clave.
# --------------------------------------------------------------------------
rf_grid <- expand.grid(
  # mtry: Número de predictores a evaluar en cada división. El valor teórico recomendado
  #       es sqrt(n_predictores) ~ 6.4. Se explora un rango alrededor de este valor.
  mtry = c(4, 7, 10, 15),
  
  # min.node.size: Tamaño mínimo del nodo terminal. Controla la complejidad de los árboles.
  #                Valores pequeños (1) crean árboles profundos; valores grandes (10)
  #                los simplifican para evitar sobreajuste.
  min.node.size = c(1, 5, 10),
  
  # splitrule: Criterio de división. Se compara el "gini" estándar con "extratrees",
  #            que añade aleatoriedad extra y puede mejorar la generalización.
  splitrule = c("gini", "extratrees")
)
cat("-> Grid de hiperparámetros para Random Forest definido.\n")


# --- 6. ENTRENAMIENTO DEL MODELO ---
# Se lanza el entrenamiento usando la función train() de caret.
# --------------------------------------------------------------------------
cat("\n*** INICIANDO ENTRENAMIENTO DEL MODELO RANDOM FOREST ***\n")
set.seed(123) # Para reproducibilidad del entrenamiento.
rf_model <- train(
  class ~ .,                  # Fórmula: 'class' es la variable a predecir, '~ .' usa todos los demás como predictores.
  data = train_set,             # Conjunto de datos de entrenamiento.
  method = "ranger",            # Algoritmo a utilizar.
  trControl = train_control,    # Estrategia de entrenamiento definida previamente.
  tuneGrid = rf_grid,           # Rejilla de hiperparámetros a probar.
  metric = "Kappa",             # Métrica a optimizar: se seleccionará la combinación que maximice Kappa.
  importance = "permutation"    # Método para calcular la importancia de variables. La permutación es más
  # robusta y fiable que la reducción de impureza (gini).
)


# --- 7. GUARDADO Y FINALIZACIÓN ---
# El modelo entrenado, que contiene los mejores hiperparámetros y toda la
# información del proceso, se guarda en un fichero .rds para su posterior análisis.
# --------------------------------------------------------------------------
cat("\n*** ENTRENAMIENTO FINALIZADO ***\n")
cat("-> Guardando el objeto del modelo en 'modelo_rf_final.rds'...\n")
saveRDS(rf_model, file = "modelo_rf_final.rds")

# Se imprimen los resultados del entrenamiento en la salida estándar.
cat("\n-> Resumen del modelo entrenado:\n")
print(rf_model)

# Se detiene el clúster de paralelización para liberar los recursos.
stopCluster(cl)
cat("\n-> Proceso completado. Clúster de paralelización detenido.\n")

