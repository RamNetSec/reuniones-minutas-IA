# Generador de Minutas de Reuniones

Este proyecto proporciona una herramienta para crear minutas estructuradas de reuniones a partir de transcripciones de texto, utilizando la API de OpenAI. Está diseñado para facilitar la síntesis de conversaciones grabadas en reuniones y convertirlas en un formato legible y estructurado.

## Características

- **Automatización completa**: Genera minutas a partir de transcripciones con un simple comando.
- **Personalizable**: Fácil de adaptar para diferentes formatos y detalles de minutas.
- **Soporte de múltiples idiomas**: Capacidad de generar minutas en varios idiomas (dependiendo de la configuración de la API de OpenAI utilizada).

## Instalación

Antes de poder ejecutar el script, necesitas instalar algunas dependencias:

```bash
pip install openai
```
# Transcriptor de Audio Automático

Este proyecto proporciona una herramienta automatizada para transcribir archivos de audio en una carpeta especificada utilizando la API de OpenAI's Whisper. La herramienta también utiliza un sistema de caché para mejorar la eficiencia al evitar la transcripción repetida de los mismos archivos.

## Características

- **Transcripción Automática**: Convierte automáticamente los archivos de audio a texto.
- **Caching**: Guarda transcripciones previas para mejorar la eficiencia en futuras ejecuciones.
- **Barra de Progreso**: Muestra el progreso de la transcripción de los archivos usando `tqdm`.
- **Logging**: Registra todos los pasos importantes del proceso para facilitar la depuración.

## Requisitos

Antes de ejecutar el script, necesitas instalar algunas dependencias:

```bash
pip install requests tqdm argparse openai logging pickle math json
```
