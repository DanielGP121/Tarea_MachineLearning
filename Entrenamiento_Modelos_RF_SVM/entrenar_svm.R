# --------------------------------------------------------------------------
# SCRIPT DE ENTRENAMIENTO PARA MÁQUINAS DE VECTORES SOPORTE (SVM)
# --------------------------------------------------------------------------
# Objetivo: Entrenar y optimizar un modelo SVM con kernel radial.
# Autor: Daniel González Palazón 
# --------------------------------------------------------------------------


# --- 1. CARGA DE LIBRERÍAS ---
# - caret: Framework principal de modelado.
# - kernlab: Proporciona la implementación del algoritmo SVM ('ksvm').
# - doParallel: Para la ejecución en paralelo.
# --------------------------------------------------------------------------
library(caret)
library(kernlab)
library(doParallel)


# --- 2. CONFIGURACIÓN DEL ENTORNO PARALELO ---
# Idéntico al script de RF para aprovechar los cores del clúster.
# --------------------------------------------------------------------------
num_cores <- as.numeric(Sys.getenv("SLURM_NTASKS", unset = 1))
if (num_cores == 1) {
  num_cores <- detectCores() - 1
}
cl <- makeCluster(num_cores)
registerDoParallel(cl)
cat(paste("-> Registrados", getDoParWorkers(), "cores para el entrenamiento en paralelo.\n"))


# --- 3. CARGA Y SUBMUESTREO DE DATOS ---
# Las SVM tienen una complejidad computacional que escala al menos cuadráticamente
# con el número de muestras (O(n²)). Entrenar con el conjunto completo de
# ~144,000 muestras sería inviable.
#
# Decisión Estratégica: Se entrenará con un subconjunto estratificado de 40,000
# muestras. Esta cifra representa un compromiso razonado entre:
#   a) Viabilidad computacional: Permite que el entrenamiento finalice en un tiempo razonable.
#   b) Representatividad: 40,000 muestras es un tamaño considerable que todavía
#      permite al modelo capturar la estructura subyacente de los datos.
# --------------------------------------------------------------------------
cat("-> Cargando el conjunto de entrenamiento completo 'train_set.rds'...\n")
train_set_full <- readRDS("train_set.rds")

cat(paste("-> Realizando submuestreo estratificado para SVM (objetivo: ~40,000 muestras) por eficiencia...\n"))
set.seed(456) # Semilla diferente para el submuestreo
# Se calcula el porcentaje necesario para obtener ~40k muestras (40000 / 144000 ≈ 0.28)
train_subset_idx <- createDataPartition(train_set_full$class, p = 0.28, list = FALSE)
train_set <- train_set_full[train_subset_idx, ]
cat(paste("-> Conjunto de entrenamiento para SVM reducido a", nrow(train_set), "muestras.\n"))


# --- 4. DEFINICIÓN DE LA ESTRATEGIA DE ENTRENAMIENTO ---
# Se utiliza exactamente la misma configuración de 'trainControl' que para el Random Forest.
# Esto es crucial para poder realizar una comparación justa y directa entre ambos modelos,
# ya que fueron evaluados bajo las mismas condiciones de remuestreo.
# --------------------------------------------------------------------------
train_control <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  verboseIter = TRUE,
  allowParallel = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  sampling = "down"
)
cat("-> Estrategia de entrenamiento definida: Idéntica a la de Random Forest para una comparación justa.\n")


# --- 5. DEFINICIÓN DEL GRID DE HIPERPARÁMETROS ---
# Se define la rejilla para los hiperparámetros de 'svmRadial'.
# --------------------------------------------------------------------------
svm_grid <- expand.grid(
  # sigma: Parámetro del kernel RBF que controla la flexibilidad de la frontera.
  #        Valores pequeños son adecuados para datos escalados. Se explora un rango
  #        para probar fronteras de decisión con diferente complejidad.
  sigma = c(0.01, 0.05, 0.1),
  
  # C (Coste): Parámetro de regularización. Controla el trade-off entre un margen
  #            de separación amplio y clasificar correctamente los puntos. Se
  #            explora en escala logarítmica.
  C = c(0.1, 1, 10)
)
cat("-> Grid de hiperparámetros para SVM definido.\n")


# --- 6. ENTRENAMIENTO DEL MODELO ---
# Se lanza el entrenamiento.
# --------------------------------------------------------------------------
cat("\n*** INICIANDO ENTRENAMIENTO DEL MODELO SVM ***\n")
set.seed(123)
svm_model <- train(
  class ~ .,
  data = train_set,
  method = "svmRadial",
  trControl = train_control,
  # preProcess: ¡Paso obligatorio para SVM! Se centra (media=0) y escala
  #             (desv.est.=1) cada predictor. Esto evita que las variables con
  #             rangos numéricos grandes dominen el modelo. Caret lo aplica de
  #             forma segura dentro de cada fold de la validación cruzada.
  preProcess = c("center", "scale"),
  tuneGrid = svm_grid,
  metric = "Kappa"
)


# --- 7. GUARDADO Y FINALIZACIÓN ---
# Se guarda el modelo final y se liberan los recursos.
# --------------------------------------------------------------------------
cat("\n*** ENTRENAMIENTO FINALIZADO ***\n")
cat("-> Guardando el objeto del modelo en 'modelo_svm_final.rds'...\n")
saveRDS(svm_model, file = "modelo_svm_final.rds")

cat("\n-> Resumen del modelo entrenado:\n")
print(svm_model)

stopCluster(cl)
cat("\n-> Proceso completado. Clúster de paralelización detenido.\n")
