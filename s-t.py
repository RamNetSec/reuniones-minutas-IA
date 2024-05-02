import os
import argparse
import tempfile
import pickle
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from openai import OpenAI
import logging

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('transcriber.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class AudioTranscriber:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.cache = {}
        self.output_folder = os.path.join(self.folder_path, 'transcriptions')
        os.makedirs(self.output_folder, exist_ok=True)
        self.load_cache()
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    def extract_audio(self, video_file_path):
        try:
            with VideoFileClip(video_file_path) as video:
                logger.info(f"Extracting audio from {video_file_path}...")
                temp_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
            return temp_audio_path
        except Exception as e:
            logger.error(f"Error processing video file {video_file_path}: {e}")
            return None

    def split_audio(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            max_size = 25 * 1000 * 1000  # 25 MB in bytes
            duration = len(audio)  # Duration in milliseconds
            frame_size = audio.frame_width * audio.frame_rate * audio.channels * (1 / 1000)  # Frame size per millisecond

            if audio.frame_count() * audio.frame_width <= max_size:
                return [file_path]  # If the size is within limit, return as is

            max_duration = max_size / frame_size  # Max duration in milliseconds that fits in 25 MB
            parts = []
            for start in range(0, duration, int(max_duration)):
                end = start + int(max_duration)
                part = audio[start:end]
                part_file_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                part.export(part_file_path, format="wav")
                parts.append(part_file_path)

            return parts
        except Exception as e:
            logger.error(f"Error splitting audio file {file_path}: {e}")
            return []

    def transcribe_audio(self, file_path):
        logger.info(f"Transcribing audio from {file_path}...")
        try:
            with open(file_path, 'rb') as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                )
            if hasattr(response, 'text'):
                return response.text
            else:
                logger.error("Response does not contain the expected text.")
                return None
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    def process_files(self):
        logger.info(f"Processing files in {self.folder_path}...")
        try:
            total_temp_files = 0
            for root, _, files in os.walk(self.folder_path):
                with tqdm(total=len(files), ncols=70, desc="Processing files") as pbar:
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.lower().endswith('.mp4'):
                            audio_path = self.extract_audio(file_path)
                            if audio_path:
                                audio_paths = self.split_audio(audio_path)
                                total_temp_files += len(audio_paths)
                                result = self.process_audio_files(audio_paths, pbar, total_temp_files)
                                self.handle_transcription(file_path, result)
                        elif file.lower().endswith(('.mp3', '.wav', '.m4a', '.mpga')):
                            audio_paths = self.split_audio(file_path)
                            total_temp_files += len(audio_paths)
                            result = self.process_audio_files(audio_paths, pbar, total_temp_files)
                            self.handle_transcription(file_path, result)
        except Exception as e:
            logger.error(f"Error processing files: {e}")

    def process_audio_files(self, audio_paths, pbar, total_temp_files):
        transcriptions = []
        for path in audio_paths:
            if path in self.cache:
                logger.info(f"Using cached transcription for {path}...")
                transcriptions.append(self.cache[path])
            else:
                transcription = self.transcribe_audio(path)
                if transcription:
                    self.cache[path] = transcription
                    transcriptions.append(transcription)
                    os.remove(path)  # Clean up the temporary audio file
                    total_temp_files -= 1
            pbar.set_postfix(temp_files=f"{total_temp_files} files")
            pbar.update()
        return " ".join(transcriptions)

    def handle_transcription(self, file_path, result):
        logger.info(f"Handling transcription for {file_path}...")
        logger.info(f"Transcription of {file_path}: {result[:50]}...")
        txt_file_path = os.path.join(self.output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt")
        with open(txt_file_path, 'w') as f:
            f.write(result + "\n")

    def load_cache(self):
        logger.info("Loading cache...")
        cache_path = os.path.join(tempfile.gettempdir(), 'transcription_cache.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.cache = pickle.load(f)

    def save_cache(self):
        logger.info("Saving cache...")
        cache_path = os.path.join(tempfile.gettempdir(), 'transcription_cache.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files in a folder to text using OpenAI's Whisper API.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing audio files.")
    args = parser.parse_args()

    transcriber = AudioTranscriber(args.folder_path)
    transcriber.process_files()
    transcriber.save_cache()
