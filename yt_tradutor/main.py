import yt_dlp
import whisper
from moviepy.editor import VideoFileClip
from transformers import pipeline
import os

# Função para download do vídeo do YouTube
def download_video(url, filename='video.mp4'):
    print(f"Iniciando o download do vídeo de {url}...")
    ydl_opts = {
        'outtmpl': filename,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4'
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Vídeo baixado e salvo como {filename}")
    except Exception as e:
        print(f"Erro ao baixar o vídeo: {e}")

# Função para extração de áudio do vídeo
def extract_audio(video_filename, audio_filename='audio.wav'):
    output_folder = os.path.join(os.getcwd(), 'yt_tradutor')
    os.makedirs(output_folder, exist_ok=True)
    audio_filepath = os.path.join(output_folder, audio_filename)
    
    try:
        print(f"Iniciando a extração do áudio de {video_filename}...")
        video = VideoFileClip(video_filename)
        
        if video.audio is None:
            print("Erro: O vídeo não contém uma faixa de áudio.")
            return None
        
        video.audio.write_audiofile(audio_filepath)
        print(f"Áudio extraído e salvo como {audio_filepath}")
        return audio_filepath
    except Exception as e:
        print(f"Erro ao extrair o áudio: {e}")
        return None

# Função para transcrição de áudio usando Whisper
def transcribe_audio(audio_filepath):
    try:
        if not os.path.isfile(audio_filepath):
            print(f"Erro: O arquivo de áudio {audio_filepath} não existe.")
            return None
        
        print(f"Transcrevendo o áudio em: {audio_filepath}")
        model = whisper.load_model("base")
        result = model.transcribe(audio_filepath)
        return result['text']
    except Exception as e:
        print(f"Erro na transcrição: {e}")
        return None

# Função para resumir o texto usando um modelo de NLP
def summarize_text(text):
    try:
        summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6")
        max_chunk_size = 1000
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        summary = []
        for chunk in chunks:
            summary_chunk = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summary.append(summary_chunk[0]['summary_text'])
        
        return " ".join(summary)
    except Exception as e:
        print(f"Erro ao resumir o texto: {e}")
        return None

# Função principal que coordena todas as etapas
def main(video_url):
    video_filename = 'video.mp4'
    audio_filename = 'audio.wav'
    
    # Baixar o vídeo
    download_video(video_url, video_filename)
    
    # Extrair áudio do vídeo baixado
    audio_filepath = extract_audio(video_filename, audio_filename)
    
    if audio_filepath:
        # Transcrever o áudio extraído
        print("Transcrevendo áudio...")
        text = transcribe_audio(audio_filepath)
        
        if text:
            # Resumir o texto transcrito
            print("Resumindo texto...")
            summary = summarize_text(text)
            
            if summary:
                print("Resumo do vídeo:")
                print(summary)
            else:
                print("Não foi possível gerar um resumo.")
        else:
            print("Não foi possível transcrever o áudio.")
    else:
        print("Não foi possível extrair o áudio.")

if __name__ == "__main__":
    video_url = input("Digite a URL do vídeo: ")
    main(video_url)
