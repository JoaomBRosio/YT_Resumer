import yt_dlp
import whisper
from moviepy.editor import VideoFileClip
from transformers import pipeline
import os

def download_video(url, filename):
    print(f"Iniciando o download do vídeo de {url}...")
    ydl_opts = {
        'outtmpl': filename,
        'format': 'mp4',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Vídeo baixado e salvo como {filename}")
    except Exception as e:
        print(f"Erro ao baixar o vídeo: {e}")

def extract_audio(video_filename, audio_filename):
    output_folder = r'C:\_Curso_Python\_TESTES EM PY\yt_tradutor'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    audio_filepath = os.path.join(output_folder, audio_filename)
    
    try:
        print(f"Iniciando a extração do áudio de {video_filename}...")
        video = VideoFileClip(video_filename)
        audio = video.audio
        audio.write_audiofile(audio_filepath)
        if os.path.exists(audio_filepath):
            print(f"Áudio extraído e salvo como {audio_filepath}")
        else:
            print(f"Erro: O arquivo de áudio {audio_filepath} não foi encontrado")
    except Exception as e:
        print(f"Erro ao extrair o áudio: {e}")

def transcribe_audio(audio_filename):
    model = whisper.load_model("base")
    try:
        absolute_path = os.path.abspath(audio_filename)
        print(f"Caminho absoluto do arquivo de áudio: {absolute_path}")

        if os.path.isfile(absolute_path):
            print(f"Transcrevendo o arquivo de áudio em: {absolute_path}")
            result = model.transcribe(absolute_path)
            return result['text']
        else:
            print(f"Erro: O arquivo de áudio {absolute_path} não existe ou não é um arquivo válido.")
    except Exception as e:
        print(f"Erro na transcrição: {e}")
        return ""

from transformers import pipeline

def summarize_text(text):
    # Define explicitamente o nome do modelo e a revisão
    summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")
    
    try:
        max_chunk_size = 1000
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        summary = []
        for chunk in chunks:
            summary_chunk = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summary.append(summary_chunk[0]['summary_text'])
        
        return " ".join(summary)
    except Exception as e:
        print(f"Erro ao resumir o texto: {e}")
        return ""

def main(video_url):
    video_filename = 'video.mp4'
    audio_filename = 'audio.wav'

    download_video(video_url, video_filename)
    extract_audio(video_filename, audio_filename)
    
    print("Transcrevendo áudio...")
    audio_filepath = os.path.join(r'C:\_Curso_Python\_TESTES EM PY\yt_tradutor', audio_filename)
    text = transcribe_audio(audio_filepath)
    
    if text:
        print("Resumindo texto...")
        summary = summarize_text(text)
        
        print("Resumo do vídeo:")
        print(summary)
    else:
        print("Não foi possível transcrever o áudio.")

if __name__ == "__main__":
    video_url = input("Digite a URL do vídeo: ")
    main(video_url)
