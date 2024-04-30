import argparse
import os
import logging
import openai
from tqdm import tqdm
import tempfile
import pickle
import math
import requests
import json

class AudioTranscriber:
    def __init__(self, folder_path, api_key):
        self.folder_path = folder_path
        self.api_key = api_key
        self.cache = {}

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def transcribe_audio(self, file_path):
        try:
            logging.info(f"Cargando y procesando el audio de {file_path}...")
            with open(file_path, 'rb') as f:
                audio = f.read()

            headers = {
                'Content-Type': 'application/x-wav',
                'Authorization': f'Bearer {self.api_key}'
            }

            response = requests.post('https://api.openai.com/v1/whisper/recognize', headers=headers, data=audio)
            response.raise_for_status()

            result = json.loads(response.text)
            return result['transcription']
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
                            if file_path in self.cache:
                                logging.info(f"Usando la transcripción en caché para {file_path}...")
                                result = self.cache[file_path]
                            else:
                                logging.info(f"Procesando el archivo {file_path}...")
                                result = self.transcribe_audio(file_path)
                                self.cache[file_path] = result
                            logging.info(f"Transcripción de {file}:")
                            logging.info(result)
                            logging.info("\n" + "-"*60 + "\n")
                        pbar.update()
        except Exception as e:
            logging.error("Error al procesar los archivos: ", e)

    def run(self):
        self.setup_logging()
        self.load_cache()
        self.process_files()
        self.save_cache()

    def load_cache(self):
        cache_path = os.path.join(tempfile.gettempdir(), 'transcription_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)

    def save_cache(self):
        cache_path = os.path.join(tempfile.gettempdir(), 'transcription_cache.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files in a folder to text using OpenAI's Whisper API.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files.")
    parser.add_argument("api_key", type=str, help="API key for OpenAI's Whisper API.")
    args = parser.parse_args()

    transcriber = AudioTranscriber(args.folder_path, args.api_key)
    transcriber.run()