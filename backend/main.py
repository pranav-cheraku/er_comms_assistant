# Main driver


# flask endpoints, call chatbot_intake class and doctor_summary class modules

# frontend info -> chatbot_intake -> summarize -> frontend

# record speech -> forward to doctor_summary module -> speech to text, extraction, text simplification -> output to frontend -> send email

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from doctor_summary import DoctorSummary

app = Flask(__name__)
CORS(app)

# Initialize DoctorSummary instance
doctor_summary = DoctorSummary()

# Set up upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio_files')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def driver():
    print('hello')

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    try:
        # Check if audio file is present in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        filename = request.form.get('filename', 'recording.webm')
        
        # Validate file extension
        allowed_extensions = {'webm', 'wav', 'mp3', 'ogg'}
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Allowed types: webm, wav, mp3, ogg'}), 400
        
        # Save the audio file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        audio_file.save(file_path)
        
        # Process the audio file using DoctorSummary
        result = doctor_summary.process_audio(file_path)
        
        # Return the results
        return jsonify({
            'transcription': result['transcription'],
            'summary': result['summary'],
        })
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-text', methods=['POST'])
def process_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Process the text using DoctorSummary
        result = doctor_summary.process_text(text)
        
        # Return the results
        return jsonify({
            'response': result['response'],
            'summary': result['summary']
        })
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
    