import argparse
import whisper
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import logging
from pydub import AudioSegment
import os

class VideoToTextConverter:
    def __init__(self, video_path, audio_path, output_path):
        self.video_path = video_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.model = whisper.load_model("small")
        logging.basicConfig(level=logging.INFO)

    def convert_video_to_audio(self):
        """Extrae el audio de un archivo de video y lo guarda como MP3."""
        try:
            video = VideoFileClip(self.video_path)
            video.audio.write_audiofile(self.audio_path, codec='mp3')
            video.close()
            logging.info("Video converted to audio successfully.")
        except Exception as e:
            logging.error(f"Error converting video to audio: {e}")
            raise

    def split_audio(self):
        """Divide el audio en segmentos de 1 minuto."""
        try:
            full_audio = AudioSegment.from_mp3(self.audio_path)
            duration = len(full_audio)
            # 60000 ms = 60 seconds = 1 minute
            return [full_audio[i:i+60000] for i in range(0, duration, 60000)]
        except Exception as e:
            logging.error(f"Error splitting the audio: {e}")
            raise

    def transcribe_audio(self):
        """Transcribe el archivo de audio y guarda el texto resultante."""
        try:
            audio_segments = self.split_audio()
            with open(self.output_path, "w") as f:
                for i, segment in enumerate(tqdm(audio_segments, desc="Transcribing")):
                    temp_file = f"temp_{i}.mp3"
                    segment.export(temp_file, format="mp3")
                    result = self.model.transcribe(temp_file)
                    f.write(result['text'] + '\n')
                    os.remove(temp_file)  # Elimina el archivo temporal justo después de usarlo
            logging.info("Audio transcription completed successfully.")
        except Exception as e:
            logging.error(f"Error transcribing the audio: {e}")
            raise

    def process(self):
        self.convert_video_to_audio()
        self.transcribe_audio()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 video to MP3 audio, split it and transcribe to text.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--audio", type=str, required=True, help="Intermediate output path for the MP3 audio file")
    parser.add_argument("--output", type=str, required=True, help="Final output path for the transcribed text")
    args = parser.parse_args()

    converter = VideoToTextConverter(args.video, args.audio, args.output)
    try:
        converter.process()
    except Exception as e:
        logging.error(f"Failed to complete the conversion and transcription process: {e}")
