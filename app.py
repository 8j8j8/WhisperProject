from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("base")  # You can change to "tiny" for faster but less accurate

@app.route('/upload', methods=['POST'])
def upload_audio():
    print("Received a POST request at /upload")

    if 'audio' not in request.files:
        print("No audio file part found in request")
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio = request.files['audio']
    print(f"Received file: {audio.filename}")

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(file_path)
    print(f"Saved file to {file_path}")

    # Transcribe audio
    result = model.transcribe(file_path)
    transcript = result.get('text', '')
    print(f"Transcript: {transcript}")

    return jsonify({
        'message': 'Audio received and transcribed',
        'transcript': transcript
    })

@app.route('/')
def home():
    return "Flask server is running! Use POST /upload to send audio."

    

if __name__ == "__main__":
    app.run(debug=True)