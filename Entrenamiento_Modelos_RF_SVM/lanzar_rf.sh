#!/bin/bash

# --------------------------------------------------------------------------
# SCRIPT DE LANZAMIENTO PARA RANDOM FOREST (SLURM)
# --------------------------------------------------------------------------

# --- DIRECTIVAS DE SLURM ---

#SBATCH -p ledley-q             # Partición o cola: Especifica en qué conjunto de nodos se ejecutará el trabajo.
#SBATCH --chdir=/home/alumno09/Tarea_MachineLearning/ # Directorio de trabajo: Es crucial para que el script encuentre los ficheros .R y .rds.
#SBATCH -J RF_DanielG           # Nombre del trabajo: Un identificador único para monitorizar el trabajo en la cola (squeue).
#SBATCH --cpus-per-task=20      # Solicitud de CPUs: Pide 20 cores para este trabajo, que serán usados por R para la paralelización.
#SBATCH --mem=32G               # Solicitud de memoria: Pide 32 Gigabytes de RAM.
#SBATCH --output=rf_training.out # Fichero para guardar la salida estándar del script (logs, prints de R).
#SBATCH --error=rf_training.err  # Fichero para guardar los errores que puedan ocurrir durante la ejecución.

# --- EJECUCIÓN DEL SCRIPT ---

# Mensaje de inicio para el log.
echo "========================================================"
echo "Lanzando trabajo de entrenamiento de Random Forest"
echo "Fecha: $(date)"
echo "Nodo: $(hostname)"
echo "Directorio: $(pwd)"
echo "CPUs asignadas: $SLURM_CPUS_PER_TASK"
echo "========================================================"

# Exportar la variable de SLURM para que el script de R la pueda leer.
# Esto permite que R sepa cuántos cores tiene disponibles para la paralelización.
export SLURM_NTASKS=$SLURM_CPUS_PER_TASK

# Ejecuta el script de R usando 'time' para medir el tiempo total de ejecución.
# Asegúrate de que el script 'entrenar_rf.R' está en el mismo directorio.
time Rscript entrenar_rf.R

echo
echo "========================================================"
echo "Trabajo de Random Forest finalizado."
echo "Fecha: $(date)"
echo "========================================================"

