from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import insightface
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Initialize InsightFace models
face_analysis = insightface.app.FaceAnalysis(name='buffalo_l')
face_analysis.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

# Set the folder for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')  # Or some other response


# Route to upload image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the uploaded image
        img = Image.open(filepath)
        img = np.array(img)

        # Perform face analysis on the uploaded image
        faces = face_analysis.get(img)
        
        if len(faces) > 0:
            return jsonify({"message": "Image processed"})
        else:
            return jsonify({"error": "No face detected in the uploaded image"})
    return jsonify({"error": "Invalid file"})

# Serve the webcam feed and perform face swapping
@app.route('/webcam_feed')
def webcam_feed():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        cap.release()
        return jsonify({"error": "Failed to capture video"})
    
    # Convert to RGB (since OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face analysis on the webcam frame
    faces_webcam = face_analysis.get(frame_rgb)

    # Check if faces are detected in the webcam frame
    if len(faces_webcam) > 0:
        # Load the uploaded image (use a sample uploaded image path or cache it from the last uploaded image)
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        img2 = cv2.imread(uploaded_image_path)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Perform face analysis on the uploaded image
        faces_uploaded = face_analysis.get(img2_rgb)

        if len(faces_uploaded) > 0:
            # Swap faces using the first detected face from the webcam and the uploaded image
            img1_ = swapper.get(frame_rgb, faces_uploaded[0], faces_webcam[0], paste_back=True)

            # Convert the swapped image back to BGR for OpenCV display
            img1_bgr = cv2.cvtColor(img1_, cv2.COLOR_RGB2BGR)

            # Encode the frame as JPEG to return in the response
            ret, jpeg = cv2.imencode('.jpg', img1_bgr)
            if not ret:
                cap.release()
                return jsonify({"error": "Failed to encode video frame"})

            # Release the webcam
            cap.release()
            return jpeg.tobytes()
        else:
            cap.release()
            return jsonify({"error": "No face detected in the uploaded image"})
    else:
        cap.release()
        return jsonify({"error": "No face detected in the webcam frame"})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=8000,debug=True)
