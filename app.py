import os
from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to preprocess the frame (e.g., convert to grayscale)
def preprocess_frame(frame):
    # Convert frame to grayscale as an example preprocessing step
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert the grayscale frame back to a color (3 channels) for display
    color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    return color_frame

# Route to serve the main page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the uploaded frame
    if 'frame' not in request.files:
        return "No frame uploaded", 400

    file = request.files['frame']
    
    # Read the frame data
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return "Invalid frame", 400

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Convert processed frame to bytes
    _, buffer = cv2.imencode('.jpg', processed_frame)
    img_bytes = buffer.tobytes()

    # Send the processed frame back as a response
    return send_file(
        BytesIO(img_bytes),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='processed_frame.jpg'
    )

if __name__ == '__main__':
    app.run(debug=True)
