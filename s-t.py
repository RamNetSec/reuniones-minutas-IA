import argparse
import os
import logging
import whisper
from tqdm import tqdm
import openai

class AudioTranscriber:
    def __init__(self, folder_path, api_key):
        self.folder_path = folder_path
        self.api_key = api_key
        self.model = None

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_model(self):
        try:
            logging.info("Cargando el modelo de Whisper...")
            self.model = whisper.load_model("base")
        except Exception as e:
            logging.error("Error al cargar el modelo de Whisper: ", e)

    def transcribe_audio(self, file_path):
        try:
            logging.info(f"Cargando y procesando el audio de {file_path}...")
            audio = whisper.load_audio(file_path)
            audio = whisper.pad_or_trim(audio)

            logging.info("Generando las predicciones...")
            mel = self.model.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            options = whisper.DecodingOptions(language="es", without_timestamps=True)
            result = self.model.decode(mel, options)
            return result.text
        except Exception as e:
            logging.error("Error al transcribir el audio: ", e)

    def process_files(self):
        try:
            logging.info(f"Recorriendo los archivos en {self.folder_path}...")
            for root, dirs, files in os.walk(self.folder_path):
                with tqdm(total=len(files), ncols=70, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                    for file in files:
                        if file.endswith('.mp3') or file.endswith('.wav'):
                            file_path = os.path.join(root, file)
                            logging.info(f"Procesando el archivo {file_path}...")
                            result = self.transcribe_audio(file_path)
                            logging.info(f"Transcripción de {file}:")
                            logging.info(result)
                            logging.info("\n" + "-"*60 + "\n")
                        pbar.update()
        except Exception as e:
            logging.error("Error al procesar los archivos: ", e)

    def run(self):
        self.setup_logging()
        self.load_model()
        self.process_files()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files in a folder to text using OpenAI's Whisper API.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files.")
    parser.add_argument("api_key", type=str, help="API key for OpenAI's Whisper API.")
    args = parser.parse_args()

    transcriber = AudioTranscriber(args.folder_path, args.api_key)
    transcriber.run()