import os
import cv2
import insightface
from flask import Flask, request, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize FaceAnalysis and Face Swapper from InsightFace
face_analysis = insightface.app.FaceAnalysis(name='buffalo_l')
face_analysis.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

# Check for allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('index.html', filename=filename)

# Route to serve uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Function to generate frames for webcam feed with face swap
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Load the uploaded image for face swapping (if available)
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'your_uploaded_image.jpg')
        if os.path.exists(uploaded_image_path):
            img2 = cv2.imread(uploaded_image_path)  # The uploaded image for face swapping

            # Perform face detection and swapping
            faces1 = face_analysis.get(frame)
            faces2 = face_analysis.get(img2)
            
            if len(faces1) > 0 and len(faces2) > 0:
                frame = swapper.get(frame, faces1[0], faces2[0], paste_back=True)
        
        # Convert frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in JPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Make sure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
