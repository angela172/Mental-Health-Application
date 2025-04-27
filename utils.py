import os
from dotenv import load_dotenv

import wave
import pyaudio
from scipy.io import wavfile
import numpy as np

import whisper

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from gtts import gTTS
import pygame


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold



def record_audio_chunk(audio, stream, chunk_length=5):
    print("Recording...")
    frames = []
    # Calculate the number of chunks needed for the specified length of recording
    # 16000 Hertz -> sufficient for capturing the human voice
    # 1024 frames -> the higher, the higher the latency
    num_chunks = int(16000 / 1024 * chunk_length)

    # Record the audio data in chunks
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path =  os.path.join(os.getcwd(), "temp_audio_chunk.wav")
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # Sample width
        wf.setframerate(16000)  # Sample rate
        wf.writeframes(b''.join(frames))  # Write audio frames

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")


def load_whisper():
    model = whisper.load_model("base")
    return model


def transcribe_audio(model, file_path):
    try:
        print("Starting transcription...")
        print(f"Checking file: {file_path}")
        if os.path.isfile(file_path):
            print(f"File {file_path} found, transcribing...")
            results = model.transcribe(file_path, fp16=False)  # Change fp16 based on your setup
            print("Transcription completed.")
            return results['text']
        else:
            print(f"File {file_path} not found.")
            return None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def load_prompt():
    input_prompt = """

    You are a mental health assistant designed to provide supportive, empathetic, and concise responses. 
    Your primary role is to detect and display the predicted emotion from the classes ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise').
    Your responses should be short, kind, and understanding. Greet the user warmly and ask how they are feeling. 
    Provide brief, helpful advice for stress management, such as breathing exercises or affirmations. 
    Be encouraging but avoid long explanations.If unsure about something, suggest professional help.

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:
    """
    return input_prompt


def load_llm():
    chat_groq = ChatGroq(temperature=0, model_name="llama3-8b-8192",
                         groq_api_key=groq_api_key)
    return chat_groq


def get_response_llm(user_question, memory):
    input_prompt = load_prompt()

    chat_groq = load_llm()

    # Look how "chat_history" is an input variable to the prompt template
    prompt = PromptTemplate.from_template(input_prompt)

    chain = LLMChain(
        llm=chat_groq,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    response = chain.invoke({"question": user_question})

    return response['text']


def play_text_to_speech(text, language='en', slow=False):
    # Generate text-to-speech audio from the provided text
    tts = gTTS(text=text, lang=language, slow=slow)

    # Save the generated audio to a temporary file
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    # Initialize the pygame mixer for audio playback
    pygame.mixer.init()

    # Load the temporary audio file into the mixer
    pygame.mixer.music.load(temp_audio_file)

    # Start playing the audio
    pygame.mixer.music.play()

    # Wait until the audio playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Control the playback speed

    # Stop the audio playback
    pygame.mixer.music.stop()

    # Clean up: Quit the pygame mixer and remove the temporary audio file
    pygame.mixer.quit()
    os.remove(temp_audio_file)