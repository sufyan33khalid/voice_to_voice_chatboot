!pip install git+https://github.com/openai/whisper.git
!pip install gradio gtts soundfile
!pip install gradio groq whisper gtts soundfile
# Set your actual Groq API key securely in the environment (replace with correct key)
os.environ["GROQ_API_KEY"] = ""

# Re-initialize the Groq client with this key setup
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    client = None
    print(f"Groq initialization error: {e}")
  
# Import libraries
import os
import whisper
import soundfile as sf
from gtts import gTTS
import gradio as gr

# Set the API key directly (replace 'YOUR_GROQ_API_KEY' with your actual key)
os.environ["GROQ_API_KEY"] = ""
# Initialize Groq client
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except ImportError:
    client = None
    print("Groq library not found. Ensure it's accessible and correctly installed.")
except Exception as e:
    client = None
    print(f"Groq initialization error: {e}")

# Load Whisper model for transcription
try:
    model = whisper.load_model("base")
except AttributeError:
    print("Error loading Whisper model. Ensure whisper library is correctly installed.")

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    audio, _ = sf.read(audio_file)
    result = model.transcribe(audio_file)
    return result["text"]

# Function to get response from Groq LLM
def get_groq_response(transcribed_text):
    if client:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": transcribed_text}],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error in Groq response: {e}"
    else:
        return "Groq API is not accessible in this environment."

# Function to convert text to speech using gTTS
def text_to_speech(text):
    tts = gTTS(text, lang="en")
    audio_path = "/content/response.mp3"
    tts.save(audio_path)
    return audio_path

# Pipeline to handle the audio input, interact with LLM, and provide an audio response
def chatbot_pipeline(audio):
    # Step 1: Transcribe audio
    transcribed_text = transcribe_audio(audio)

    # Step 2: Get response from Groq LLM
    response_text = get_groq_response(transcribed_text)

    # Step 3: Convert response to speech
    response_audio = text_to_speech(response_text)

    return transcribed_text, response_text, response_audio

# Gradio interface setup
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Textbox(label="LLM Response"),
        gr.Audio(label="Response Audio")
    ],
    live=True
)

# Launch the Gradio app
iface.launch()
