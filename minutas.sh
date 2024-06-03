#!/bin/bash

# Establecer "strict mode" para mejor manejo de errores
set -euo pipefail
trap "echo 'Error: Script falló en la línea $LINENO'; exit 1" ERR

# Función para mostrar el uso del script
usage() {
    echo "Uso: $0 [-i input_dir] [-o output_dir] [-m model_size] [-l language] [-t temperature] [-d download_id] [-h]"
    echo "  -i input_dir    Directorio que contiene los archivos de video (default: videos)"
    echo "  -o output_dir   Directorio donde se guardarán las transcripciones (default: output)"
    echo "  -m model_size   Tamaño del modelo Whisper a usar (default: medium)"
    echo "  -l language     Idioma para la transcripción (default: Spanish)"
    echo "  -t temperature  Temperatura para la transcripción (default: 0.2)"
    echo "  -d download_id  ID del archivo o carpeta para descargar los videos"
    echo "  -h              Mostrar esta ayuda"
    exit 1
}

# Valores por defecto
input_dir="videos"
output_dir="output"
model_size="medium"
language="Spanish"
while getopts "i:o:m:l:t:d:h" opt; do
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
        d)
            download_id=$OPTARG
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

# Actualizar y mejorar el sistema
echo "Actualizando y mejorando el sistema..."
apt update && apt upgrade -y

# Instalar herramientas necesarias
echo "Instalando herramientas necesarias..."
apt install -y bashtop nvtop ffmpeg unzip python3-venv bc

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

# Descargar el archivo de video
if [[ -n "$download_id" ]]; then
    echo "Descargando archivo de video..."
    gdown "$download_id" -O . || { echo "Error al descargar el video."; exit 1; }
else
    echo "Error: El ID de descarga no fue proporcionado."
    exit 1
fi

# Verificar si solo hay un archivo y multiplicarlo por 10
files=(*.mp4)
if [ ${#files[@]} -eq 1 ]; then
    echo "Solo se detectó un archivo. Multiplicando por 20..."
    for i in {1..20}; do
        cp "${files[0]}" "${files[0]%.mp4}_copy_$i.mp4"
    done
    files=(*_copy_*.mp4)
fi

# Descargar el modelo Whisper antes de usarlo
echo "Descargando el modelo Whisper..."
python3 -c "import whisper; whisper.load_model('$model_size')" || { echo "Error al descargar el modelo Whisper."; exit 1; }

# Crear las carpetas de salida si no existen
echo "Creando carpetas de salida..."
mkdir -p "$output_dir/transcriptions"
mkdir -p "$output_dir/times"

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
num_files=${#files[@]}
echo "Número de archivos a procesar: $num_files"

# Función para procesar un archivo en una GPU
process_file() {
    local gpu=$1
    local file=$2
    local output_dir=$3
    
    echo "Procesando $file en GPU $gpu"
    
    # Crear carpetas de salida si no existen
    mkdir -p "$output_dir/transcriptions"
    mkdir -p "$output_dir/times"
    
    # Obtener la duración del video en segundos
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    echo "Duración del video: $duration segundos"
    
    # Iniciar temporizador
    start=$(date +%s.%N)
    
    # Procesar el archivo
    CUDA_VISIBLE_DEVICES=$gpu whisper "$file" --model $model_size --device cuda --language $language --output_dir "$output_dir/transcriptions" --fp16 True --temperature $temperature --task transcribe 2>> "$output_dir/times/time_${file%.mp4}.txt" || { echo "Error procesando el archivo $file en GPU $gpu"; }
    
    # Calcular tiempo total
    end=$(date +%s.%N)
    processing_time=$(echo "$end - $start" | bc || echo "bc no está disponible para cálculos de tiempo")
    
    # Agregar tiempo total y duración del video al archivo de tiempos
    {
        echo "Tiempo de procesamiento: $processing_time segundos"
        echo "Duración del video: $duration segundos"
    } >> "$output_dir/times/time_${file%.mp4}.txt"
}

export -f process_file
export model_size language temperature output_dir GPUS

# Calcular el número de archivos a procesar simultáneamente por GPU
files_per_gpu=$((num_files < num_gpus * 4 ? num_files / num_gpus : 4))
if [[ $files_per_gpu -eq 0 ]]; then
    files_per_gpu=1
fi
echo "Número de archivos a procesar simultáneamente por GPU: $files_per_gpu"

# Crear los comandos a ejecutar
commands=()
for ((i = 0; i < num_files; i++)); do
    gpu_index=$((i / files_per_gpu % num_gpus))
    commands+=("process_file ${GPUS[$gpu_index]} ${files[$i]} $output_dir")
done

# Mostrar comandos para depuración
echo "Comandos a ejecutar:"
for cmd in "${commands[@]}"; do
    echo "$cmd"
done

# Ejecutar comandos en paralelo usando xargs
printf "%s\n" "${commands[@]}" | xargs -P $((num_gpus * files_per_gpu)) -n 1 -I {} bash -c "{}"

echo "Proceso completado."

# Desactivar el entorno virtual
deactivate

