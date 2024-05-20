#!/bin/bash

# Establecer "strict mode" para mejor manejo de errores
set -euo pipefail
trap "echo 'Error: Script falló en la línea $LINENO'; exit 1" ERR

# Función para mostrar el uso del script
usage() {
    echo "Uso: $0 [-i input_dir] [-o output_dir] [-m model_size] [-l language] [-t temperature] [-h]"
    echo "  -i input_dir    Directorio que contiene los archivos de video (default: videos)"
    echo "  -o output_dir   Directorio donde se guardarán las transcripciones (default: output)"
    echo "  -m model_size   Tamaño del modelo Whisper a usar (default: medium)"
    echo "  -l language     Idioma para la transcripción (default: Spanish)"
    echo "  -t temperature  Temperatura para la transcripción (default: 0.7)"
    echo "  -h              Mostrar esta ayuda"
    exit 1
}

# Valores por defecto
input_dir="videos"
output_dir="output"
model_size="medium"
language="Spanish"
temperature="0.7"

# Procesar los argumentos de la línea de comandos
while getopts "i:o:m:l:t:h" opt; do
    case ${opt} in
        i)
            input_dir=$OPTARG
            ;;
        o)
            output_dir=$OPTARG
            ;;
        m)
            model_size=$OPTARG
            ;;
        l)
            language=$OPTARG
            ;;
        t)
            temperature=$OPTARG
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

# Comprobar la existencia de comandos necesarios
for cmd in apt pip3 gdown whisper nvidia-smi; do
    command -v $cmd >/dev/null 2>&1 || { echo "Error: El comando $cmd no está instalado."; exit 1; }
done

# Actualizar y mejorar el sistema
echo "Actualizando y mejorando el sistema..."
apt update && apt upgrade -y

# Instalar herramientas necesarias
echo "Instalando bashtop, nvtop, ffmpeg y unzip..."
apt install -y bashtop nvtop ffmpeg unzip python3-venv

# Crear y activar un entorno virtual de Python
echo "Creando entorno virtual de Python..."
python3 -m venv venv
source venv/bin/activate

# Instalar las bibliotecas de Python necesarias
echo "Actualizando pip e instalando bibliotecas de Python necesarias..."
pip install --upgrade pip
pip install gdown openai-whisper

# Crear y moverse al directorio de entrada
echo "Creando y moviéndose al directorio de entrada '$input_dir'..."
mkdir -p "$input_dir" && cd "$input_dir"

# Descargar un archivo zip que contiene los 100 archivos de video
echo "Descargando archivos de video..."
gdown 1WlRCDDJN8b1_HU50ygJ_zFT8xfmkdUcb --folder -O . || { echo "Error al descargar los videos."; exit 1; }

# Descargar el modelo Whisper antes de usarlo
echo "Descargando el modelo Whisper..."
python3 -c "import whisper; whisper.load_model('$model_size')" || { echo "Error al descargar el modelo Whisper."; exit 1; }

# Crear las carpetas de salida si no existen
echo "Creando carpetas de salida..."
mkdir -p "$output_dir/transcriptions" "$output_dir/times"

# Contar el número de GPUs disponibles
num_gpus=$(nvidia-smi -L | wc -l)
if [[ $num_gpus -eq 0 ]]; then
    echo "No hay GPUs disponibles"
    exit 1
fi
echo "Número de GPUs disponibles: $num_gpus"

# Lista de GPUs disponibles
GPUS=($(seq 0 $((num_gpus - 1))))

# Lista de archivos a procesar
files=(*.mp4)
num_files=${#files[@]}

# Función para procesar un archivo en una GPU
process_file() {
    local gpu=$1
    local file=$2
    
    echo "Procesando $file en GPU $gpu"
    { time CUDA_VISIBLE_DEVICES=$gpu whisper "$file" --model $model_size --device cuda --language $language --output_dir "$output_dir/transcriptions" --fp16 True --temperature $temperature --task transcribe; } 2>> "$output_dir/times/time_${file%.mp4}.txt" || { echo "Error procesando el archivo $file en GPU $gpu"; }
}

# Función para controlar la cantidad de procesos concurrentes
parallel_process() {
    local max_jobs=$1
    local current_jobs=0

    shift
    for cmd in "$@"; do
        ((current_jobs++))
        eval "$cmd" &
        if ((current_jobs >= max_jobs)); then
            wait -n
            ((current_jobs--))
        fi
    done
    wait
}

# Procesar archivos en lotes, cada GPU procesa 2 archivos por ronda
commands=()
for ((i = 0; i < num_files; i += num_gpus * 2)); do
    for ((j = 0; j < num_gpus; j++)); do
        for ((k = 0; k < 2; k++)); do
            index=$((i + j * 2 + k))
            if [ $index -lt $num_files ]; then
                commands+=("process_file '${GPUS[$j]}' '${files[$index]}'")
            fi
        done
    done
done

# Ejecutar comandos en paralelo controlado
parallel_process $num_gpus "${commands[@]}"

echo "Proceso completado."

# Desactivar el entorno virtual
deactivate

