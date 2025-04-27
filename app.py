import os
import pyaudio
import streamlit as st
from langchain.memory import ConversationBufferMemory
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import pickle
from audio_app import get_features

from utils import record_audio_chunk, transcribe_audio, get_response_llm, play_text_to_speech, load_whisper

chunk_file = r'C:\Users\11 PrO\Desktop\fds\Mental-Health-Application\src\temp_audio_chunk.wav'
model = load_whisper()
fds_model = tf.keras.models.load_model(r'C:\Users\11 PrO\Desktop\fds\my_good_model.h5')
label_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(audio_file):
    """Predict the emotion from an audio file."""
    features = get_features(audio_file)
    features = np.expand_dims(features, axis=2)  # Reshape for the model input

    # Ensure consistent feature length
    if features.shape[1] < 4536:
        features = tf.keras.utils.pad_sequences(features, maxlen=4536, padding='post', dtype='float32')
    elif features.shape[1] > 4536:
        features = features[:, :4536, :]

    # Make predictions
    predictions = fds_model.predict(features)
    predictions_mean = np.mean(predictions, axis=0)
    predicted_label = label_classes[np.argmax(predictions_mean)]
    return predicted_label

def main():
    st.markdown('<h1>Mental Health Assistant️</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:18px;">Your personal AI mental health assistant, ready to engage in supportive and empathetic conversations.</p>', unsafe_allow_html=True)

    memory = ConversationBufferMemory(memory_key="chat_history")
    
    
    if st.button("Start Recording"):
        while True:
            st.write("Recording...")
            # Audio Stream Initialization
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

            # Record and save audio chunk
            record_audio_chunk(audio, stream)

            text = transcribe_audio(model, chunk_file)

            if text is not None:
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">User 👤: {text}</div>',
                    unsafe_allow_html=True)
                
                emotion = predict_emotion(chunk_file)
                st.write(f"Predicted Emotion: {emotion}")
                os.remove(chunk_file)
                
    
                response_llm = get_response_llm(user_question=text, memory=memory)
                st.markdown(
                    f'<div style="background-color: #d6d4d4; padding: 10px; border-radius: 5px;">Mental Health Assistant 🤖: {response_llm}</div>',
                    unsafe_allow_html=True)

                play_text_to_speech(text=response_llm)
            else:
                stream.stop_stream()
                stream.close()
                audio.terminate()
                break  # Exit the while loop
        print("End Conversation")
        st.write("End of Conversation")



if __name__ == "__main__":
    main()