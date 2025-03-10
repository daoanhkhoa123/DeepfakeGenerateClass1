import os
from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
import cv2
import numpy as np
import insightface
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

# Set up upload folder for images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Make sure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize face analysis model
face_analysis = insightface.app.FaceAnalysis(name='buffalo_l')
face_analysis.prepare(ctx_id=0, det_size=(640, 640))

# Load the face-swapping model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

# Function to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to serve the main page (index.html)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/processed', methods=['POST'])
def process_frame():
    # Get the uploaded frame from the webcam
    if 'frame' not in request.files:
        return "No frame uploaded", 400  # No frame in request, return error

    file = request.files['frame']
    
    # Read the frame data
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return "Invalid frame", 400  # If frame is invalid, return error

    # Detect faces in the frame
    faces = face_analysis.get(frame)

    if len(faces) > 0:
        # Take the first detected face for swapping (this can be adjusted as needed)
        face = faces[0]
        swapped_frame = swap_face(frame, face)
    else:
        swapped_frame = frame  # If no faces detected, return the original frame

    # Convert processed frame to bytes
    _, buffer = cv2.imencode('.jpg', swapped_frame)
    img_bytes = buffer.tobytes()

    # Log the size of the image being returned
    print(f"Processed frame size: {len(img_bytes)} bytes")

    # Send the processed frame back as a response
    return send_file(
        BytesIO(img_bytes),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='swapped_frame.jpg'
    )


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = "uploaded_image.jpg"  # Use a fixed name for the uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Return the image URL so it can be displayed on the frontend
        image_url = f"/uploads/{filename}"
        return jsonify({'image_url': image_url})

    return jsonify({'error': 'File type not allowed'}), 400

# Function to swap faces
def swap_face(frame, face):
    # Extract face from uploaded image (e.g., a predefined image or webcam frame)
    uploaded_image = cv2.imread("uploads/uploaded_image.jpg")  # Fixed image name

    # Detect faces in the uploaded image
    uploaded_faces = face_analysis.get(uploaded_image)
    if len(uploaded_faces) == 0:
        return frame  # If no face detected in uploaded image, return the original frame
    
    uploaded_face = uploaded_faces[0]

    # Swap the detected face from the uploaded image with the webcam frame
    swapped_frame = swapper.get(face, uploaded_face, frame)

    return swapped_frame

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=8015, debug=True)
