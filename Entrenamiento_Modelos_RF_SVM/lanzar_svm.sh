#!/bin/bash

# --------------------------------------------------------------------------
# SCRIPT DE LANZAMIENTO PARA SVM (SLURM)
# --------------------------------------------------------------------------

# --- DIRECTIVAS DE SLURM ---

#SBATCH -p ledley-q             # Cola: Especifica en qué conjunto de nodos se ejecutará el trabajo.
#SBATCH --chdir=/home/alumno09/Tarea_MachineLearning/ # Directorio de trabajo (debe ser el correcto).
#SBATCH -J SVM_DanielG          # Nombre del trabajo, diferente al de RF para distinguirlos.
#SBATCH --cpus-per-task=20      # Solicitud de CPUs: 20 cores.
#SBATCH --mem=32G               # Solicitud de memoria: Pide 32 Gigabytes de RAM.
#SBATCH --output=svm_training.out # Fichero para guardar la salida estándar del script (logs, prints de R).
#SBATCH --error=svm_training.err  # Fichero para guardar los errores que puedan ocurrir durante la ejecución.

# --- EJECUCIÓN DEL SCRIPT ---

# Mensaje de inicio para el log.
echo "========================================================"
echo "Lanzando trabajo de entrenamiento de SVM"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "Directorio: $(pwd)"
echo "CPUs asignadas: $SLURM_CPUS_PER_TASK"
echo "========================================================"
echo

# Exportar la variable de SLURM para que R la pueda leer.
export SLURM_NTASKS=$SLURM_CPUS_PER_TASK

# Ejecuta el script de R para SVM.
# Asegúrate de que el script 'entrenar_svm.R' está en el mismo directorio.
time Rscript entrenar_svm.R

echo
echo "========================================================"
echo "Trabajo de SVM finalizado."
echo "Fecha: $(date)"
echo "========================================================"