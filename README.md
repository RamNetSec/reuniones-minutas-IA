
# Video Transcription Script

Este script automatiza la transcripción de videos utilizando el modelo Whisper de OpenAI y distribuye la carga de trabajo en múltiples GPUs disponibles. 

## Descripción

El script realiza las siguientes acciones:

1. Comprueba la existencia de comandos necesarios.
2. Actualiza y mejora el sistema.
3. Instala herramientas necesarias (bashtop, nvtop, ffmpeg, unzip, python3-venv).
4. Crea y activa un entorno virtual de Python.
5. Instala bibliotecas de Python necesarias (gdown, openai-whisper).
6. Crea y se mueve al directorio de entrada especificado.
7. Descarga archivos de video desde una URL especificada.
8. Descarga el modelo Whisper antes de usarlo.
9. Crea las carpetas de salida si no existen.
10. Cuenta el número de GPUs disponibles.
11. Distribuye el procesamiento de archivos de video entre las GPUs disponibles.
12. Realiza la transcripción de los videos y guarda los resultados en las carpetas de salida especificadas.

## Uso

### Prerrequisitos

Asegúrate de tener los siguientes comandos instalados en tu sistema:

- `apt`
- `pip3`
- `gdown`
- `whisper`
- `nvidia-smi`

### Instrucciones

1. Clona el repositorio a tu máquina local:

    ```sh
    git clone https://github.com/tuusuario/nombre-repositorio.git
    cd nombre-repositorio
    ```

2. Haz que el script sea ejecutable:

    ```sh
    chmod +x minutas.sh
    ```

3. Ejecuta el script con los argumentos necesarios:

    ```sh
    ./minutas.sh -i <input_dir> -o <output_dir> -m <model_size> -l <language> -t <temperature>
    ```

    - `-i <input_dir>`: Directorio que contiene los archivos de video (default: `videos`).
    - `-o <output_dir>`: Directorio donde se guardarán las transcripciones (default: `output`).
    - `-m <model_size>`: Tamaño del modelo Whisper a usar (default: `medium`).
    - `-l <language>`: Idioma para la transcripción (default: `Spanish`).
    - `-t <temperature>`: Temperatura para la transcripción (default: `0.7`).

### Ejemplo de Uso

```sh
./minutas.sh -i videos -o output -m large -l English -t 0.8
```

Este comando procesará los archivos de video en el directorio `videos`, guardará las transcripciones en el directorio `output`, usará el modelo Whisper `large`, transcribirá en inglés y utilizará una temperatura de `0.8`.

## Notas

- Asegúrate de que las GPUs estén disponibles y correctamente configuradas en tu sistema.
- El script crea y utiliza un entorno virtual de Python para asegurar que las dependencias necesarias estén aisladas y no interfieran con otras configuraciones de Python en tu sistema.

## Contribución

Las contribuciones son bienvenidas. Por favor, envía un pull request o abre un issue para discutir los cambios que deseas realizar.

