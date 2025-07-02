from flask import Flask
from flask_socketio import SocketIO, emit
from faster_whisper import WhisperModel
import numpy as np
import queue
import threading
import torch
from datetime import datetime
from scipy.signal import butter, filtfilt
import noisereduce as nr

app = Flask(__name__)
app.static_folder = 'static'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize faster-whisper model
print("Loading faster-whisper model...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"  # Use int8 for CPU for better performance
    # Use 'base' model for better accuracy
    model = WhisperModel("base", device=device, compute_type=compute_type)
    print(f"faster-whisper model loaded successfully (device={device}, compute_type={compute_type})")
except Exception as e:
    print(f"Error loading faster-whisper model: {e}")
    raise

# Audio settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz
audio_queue = queue.Queue(maxsize=10)  # Limit queue size
MIN_SEGMENT_LENGTH = int(SAMPLE_RATE * 0.5)  # 0.5 seconds
OVERLAP_DURATION = int(SAMPLE_RATE * 0.2)  # 200ms overlap

# Global variables
recording = False
transcription_thread_instance = None
last_audio_buffer = None  # For audio overlap
speech_buffer = []
total_samples = 0
full_transcript = ""

def bandpass_filter(audio_data, lowcut=100, highcut=4000, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio_data)

def transcribe_segment(audio_data):
    """Transcribe an audio segment using faster-whisper."""
    global last_audio_buffer
    try:
        print("\nStarting transcription process...")
        print(f"Audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        print(f"Audio stats - min: {np.min(audio_data)}, max: {np.max(audio_data)}, mean: {np.mean(audio_data)}")
        
        # Validate audio data
        if len(audio_data) < MIN_SEGMENT_LENGTH:
            print(f"Audio segment too short: {len(audio_data)} samples")
            return ""
            
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        else:
            audio_data = audio_data.astype(np.float32, copy=False)
        
        # Check if audio is too quiet
        if np.max(np.abs(audio_data)) < 0.005:
            print("Audio segment too quiet")
            return ""
            
        # Apply overlap with previous segment if available
        if last_audio_buffer is not None:
            audio_data = np.concatenate([last_audio_buffer, audio_data])
            
        # Store the end of current segment for next overlap
        last_audio_buffer = audio_data[-OVERLAP_DURATION:]
        
        # Transcribe directly from NumPy array
        print("Starting faster-whisper transcription...")
        segments, _ = model.transcribe(
            audio_data,
            language="en",  # Set to None for auto-detection if needed
            beam_size=3,  # Balance speed and accuracy
            vad_filter=True,  # Built-in VAD for silence filtering
            vad_parameters=dict(min_silence_duration_ms=700),  # Stricter silence detection
            condition_on_previous_text=False
        )
        transcript = "".join(segment.text for segment in segments).strip()
        # Filter out very short or repeated segments
        if len(transcript.split()) < 2:
            print("Transcript too short, ignoring.")
            return ""
        if hasattr(transcribe_segment, 'last_transcript') and transcript == transcribe_segment.last_transcript:
            print("Duplicate transcript, ignoring.")
            return ""
        transcribe_segment.last_transcript = transcript
        if transcript:
            print(f"Successfully transcribed: {transcript}")
        else:
            print("Empty transcription result")
        return transcript
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return ""

def transcription_thread():
    """Process audio segments from queue and emit transcripts."""
    print("Starting transcription thread...")
    while recording:
        try:
            print("Waiting for audio segment...")
            audio_data = audio_queue.get(timeout=1)
            print(f"Processing segment of length {len(audio_data)} samples")
            
            if not isinstance(audio_data, (list, np.ndarray)):
                print("Invalid audio data type")
                continue
                
            transcript = transcribe_segment(audio_data)
            if transcript:
                print(f"Emitting transcript: {transcript}")
                socketio.emit('transcript', {'transcript': transcript})
            else:
                print("Empty transcription result")
                socketio.emit('status', {'message': 'No speech detected in segment'})
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Thread error: {e}")
            import traceback
            traceback.print_exc()
            socketio.emit('status', {'message': f'Processing error: {str(e)}'})

@socketio.on('audio_segment')
def handle_audio_segment(data):
    """Receive audio segment from client."""
    global speech_buffer, total_samples
    try:
        print("\nReceived audio segment from client")
        if not isinstance(data, (list, np.ndarray)):
            print("Invalid audio data type received")
            return
            
        audio_data = np.array(data, dtype=np.float32)
        print(f"Received audio segment: {len(audio_data)} samples")
        print(f"Max amplitude before normalization: {np.max(np.abs(audio_data))}")
        
        # Validate audio length
        if len(audio_data) < MIN_SEGMENT_LENGTH:
            print("Segment too short, ignoring")
            return
        
        # Normalize audio
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            audio_data = np.clip(audio_data * 1.2, -1.0, 1.0)
        else:
            print("All samples are zero, ignoring")
            return
            
        # Apply bandpass filter and noise reduction
        audio_data = bandpass_filter(audio_data)
        audio_data = nr.reduce_noise(y=audio_data, sr=SAMPLE_RATE)
        
        # Dynamic segment length with speech buffer
        speech_buffer.append(audio_data)
        total_samples += len(audio_data)
        if total_samples >= SAMPLE_RATE * 2:  # 2 seconds
            try:
                audio_queue.put_nowait(np.concatenate(speech_buffer))
                speech_buffer.clear()
                total_samples = 0
            except queue.Full:
                print("Queue full, dropping oldest segment")
                audio_queue.get()
                audio_queue.put_nowait(np.concatenate(speech_buffer))
                speech_buffer.clear()
                total_samples = 0
                
    except Exception as e:
        print(f"Error handling audio segment: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('status', {'message': f'Error processing audio: {str(e)}'})

@socketio.on('start_recording')
def start_recording():
    """Start transcription thread."""
    global recording, transcription_thread_instance, last_audio_buffer, speech_buffer, total_samples
    print("\nStarting recording...")
    if not recording:
        recording = True
        last_audio_buffer = None
        speech_buffer.clear()
        total_samples = 0
        transcription_thread_instance = threading.Thread(target=transcription_thread)
        transcription_thread_instance.start()
        emit('status', {'message': 'Recording started'})
        print("Recording started successfully")
    else:
        emit('status', {'message': 'Recording already in progress'})
        print("Recording already in progress")

@socketio.on('stop_recording')
def stop_recording():
    """Stop transcription."""
    global recording, transcription_thread_instance
    print("\nStopping recording...")
    if recording:
        recording = False
        if transcription_thread_instance:
            transcription_thread_instance.join()
        emit('status', {'message': 'Recording stopped'})
        print("Recording stopped successfully")
    else:
        emit('status', {'message': 'No recording in progress'})
        print("No recording in progress")

@socketio.on('clear_transcript')
def clear_transcript():
    global speech_buffer, total_samples, last_audio_buffer, full_transcript
    print("\nClearing transcript and buffers...")
    speech_buffer.clear()
    total_samples = 0
    last_audio_buffer = None
    full_transcript = ""
    emit('status', {'message': 'Transcript cleared'})

@app.route('/')
def home():
    return "Flask WebSocket server is running! Connect via WebSocket for live transcription."

@app.route('/client')
def serve_client():
    return app.send_static_file('client.html')

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)