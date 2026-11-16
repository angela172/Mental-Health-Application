# FDS — FrienDShip

A voice-based mental health assistant prototype built with Streamlit. FDS records speech, transcribes it with OpenAI Whisper, predicts an emotion using TensorFlow/Keras, and generates a supportive response through Groq. Responses appear on screen and are read aloud using Google Text-to-Speech (gTTS).

This educational prototype is not a clinically validated assessment tool or a replacement for professional mental health care.

## Features

- Microphone recording in approximately five-second chunks.
- Local speech-to-text using Whisper's `base` model.
- Classification into seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise.
- AI replies through Groq and LangChain with in-memory conversation history.
- Spoken responses through gTTS and Pygame.
- An optional WAV/MP3 upload interface and a notebook of training experiments.

## How it works

```text
Microphone → WAV → Whisper → Transcript → Groq → Reply → gTTS → Speakers
              └→ Audio features → Keras model → Displayed emotion
```

Audio features include zero-crossing rate, root-mean-square energy, and MFCCs. The classifier averages predictions over the original features and noise/pitch-augmented variants.

The predicted emotion is displayed separately. The current implementation sends the transcript and conversation history to the language model, but does not pass the classifier's prediction into its prompt.

## Project structure

```text
Mental-heath-app/
├── app.py              # Main Streamlit voice conversation interface
├── audio_app.py        # Feature extraction, emotion model, and upload() UI
├── utils.py            # Recording, Whisper, Groq, and speech playback
├── mental.ipynb        # Model training and audio-processing experiments
├── requirements.txt    # Partial application dependency list
├── Dockerfile          # Basic Python 3.9 container definition
├── __init__.py
└── README.md
```

## Prerequisites

- Python and pip. The Dockerfile uses Python 3.9; the repository does not provide a tested Python/package compatibility matrix.
- A microphone and speakers on the machine running Python.
- The FFmpeg executable installed and available on `PATH`.
- Working PyAudio/PortAudio support for your operating system.
- A Groq API key and internet access for Groq, gTTS, and the initial Whisper model download.
- A compatible trained emotion model. **The referenced `my_good_model.h5` file is not included.**

**Setup status:** this repository requires local configuration before it can run. Model and recording paths are machine-specific, some imported dependencies are missing from `requirements.txt`, and package versions are not pinned.

## Setup

Run commands from the project root.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install tensorflow numpy librosa scikit-learn pygame
```

The last command supplies packages imported by the application but absent from `requirements.txt`. Notebook experiments require additional dependencies.

Verify the system FFmpeg installation:

```bash
ffmpeg -version
```

Installing `ffmpeg-python` alone does not install the FFmpeg executable. If PyAudio installation fails, install your platform's PortAudio development prerequisites or use a compatible Python/PyAudio wheel.

**LangChain compatibility:** the code imports legacy APIs such as `langchain.memory.ConversationBufferMemory` and `langchain.chains.llm.LLMChain`. An arbitrary installation of the latest packages is not guaranteed to work. Use mutually compatible LangChain and `langchain-groq` versions that expose these APIs, or migrate the code. A verified dependency lockfile is not included.

### 3. Configure Groq

Create `.env` in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

`utils.py` loads it using `python-dotenv`. `.env` is already ignored by Git; do not commit API keys.

Review `model_name` in `utils.py`'s `load_llm()`. The code currently specifies `llama3-8b-8192`. If the provider rejects that identifier, replace it with an available chat model supported by your account and installed integration.

### 4. Configure model and recording paths

Place a compatible trained model at, for example:

```text
models/my_good_model.h5
```

Replace the hardcoded path in the `tf.keras.models.load_model(...)` calls in **both `app.py` and `audio_app.py`**:

```python
tf.keras.models.load_model("models/my_good_model.h5")
```

Both modules load a model at import time, so both paths must be updated. The application prepares inputs with 4,536 feature positions and one channel. The model's output order must match:

```python
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
```

Use a model trained with compatible preprocessing. `mental.ipynb` contains experiments, but relies on external datasets and local paths; it is not a self-contained training pipeline. Its exported models should not be assumed to match the application's expected model without checking features, shapes, and labels.

In `app.py`, replace the hardcoded `chunk_file` assignment with:

```python
chunk_file = os.path.join(os.getcwd(), "temp_audio_chunk.wav")
```

This matches the location used by `record_audio_chunk()` in `utils.py`. Launch from the project root so relative paths resolve consistently.

### 5. Run the app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

1. Click **Start Recording** and speak into the microphone.
2. Wait for the recording to be transcribed and classified.
3. Read the transcript, predicted emotion, and assistant response.
4. Listen to the spoken reply; recording continues in subsequent iterations.

The loop ends when transcription returns `None`, including when silence detection removes the recording. There is no dedicated Stop button. Press `Ctrl+C` in the terminal to stop the server.

Microphone capture and playback occur on the machine running Python, not through the browser. Opening a remotely hosted page does not capture the visitor's microphone.

## Audio-upload interface

`audio_app.py` defines `upload()` but does not call it automatically. Running `streamlit run audio_app.py` alone will not display the upload interface.

Create `upload_app.py` in the project root:

```python
from audio_app import upload

upload()
```

Then run:

```bash
streamlit run upload_app.py
```

The same model and dependency prerequisites apply. The interface accepts WAV and MP3 files, plays the uploaded audio, and displays a predicted emotion. Uploaded bytes are saved locally as `temp_audio_file.wav`, including when the original upload is MP3.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Model file not found | Supply the trained `.h5` file and update both model loaders. |
| Transcription cannot find the recording | Make `app.py`'s `chunk_file` match the working-directory path used by `utils.py`. |
| Missing Python module | Install the supplemental dependencies above and check package compatibility. |
| LangChain import error | Use compatible versions exposing the legacy APIs, or update the integration. |
| FFmpeg error | Confirm `ffmpeg -version` works in the terminal launching Streamlit. |
| Groq authentication or model error | Check `.env`, `GROQ_API_KEY`, and the configured model identifier. |
| Microphone/playback error | Check OS permissions, default devices, and PyAudio/PortAudio support. |
| Conversation ends immediately | Check the recording path and microphone level; silence detection uses a fixed amplitude threshold of `3000` in `utils.py`. |

## Limitations

- Emotion labels are model predictions, not diagnoses. No validated accuracy or clinical performance is documented here.
- Random noise and pitch augmentation can cause predictions to vary across runs.
- Conversation memory is created inside `main()` and is not persisted across Streamlit reruns or application restarts.
- Recording uses a blocking loop. Audio resources are closed in the termination branch rather than after every successful iteration.
- Shared temporary filenames can cause interference between concurrent sessions.
- The Dockerfile is a starting point: it does not install all Python/system dependencies, supply the trained model, correct local paths, or configure host audio devices. It is not a ready-to-run deployment configuration.

## Data handling

Whisper transcription and emotion inference run locally. Transcribed messages and conversation history are sent to Groq. Generated reply text is sent to the service used by gTTS for speech synthesis. LangChain uses `verbose=True`, so conversation content can also appear in terminal logs.

The app writes temporary audio to disk. Normal conversation processing deletes recorded chunks and generated speech, but errors can leave files behind. The upload interface does not automatically delete `temp_audio_file.wav`. Consider these behaviors before using sensitive recordings or sharing logs.
