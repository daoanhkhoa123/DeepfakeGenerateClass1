import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
import insightface
from insightface.model_zoo import get_model

app = Flask(__name__)

# Initialize face analysis and the swapper model
face_analysis = insightface.app.FaceAnalysis(name='buffalo_l')
face_analysis.prepare(ctx_id=0, det_size=(640, 640))

swapper = get_model('inswapper_128.onnx', download=False, download_zip=False)

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to process webcam frame and swap face with the uploaded image
@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the image data from the request
    data = request.get_json()
    webcam_frame_data = data['image']
    uploaded_face_data = data['uploaded_face']

    # Decode the base64 string for the webcam frame and uploaded face
    webcam_img_data = base64.b64decode(webcam_frame_data.split(',')[1])
    uploaded_img_data = base64.b64decode(uploaded_face_data.split(',')[1])

    # Convert to numpy arrays for OpenCV processing
    np_arr_webcam = np.frombuffer(webcam_img_data, np.uint8)
    webcam_img = cv2.imdecode(np_arr_webcam, cv2.IMREAD_COLOR)
    
    np_arr_uploaded = np.frombuffer(uploaded_img_data, np.uint8)
    uploaded_img = cv2.imdecode(np_arr_uploaded, cv2.IMREAD_COLOR)

    # Perform face swapping between the webcam image and the uploaded face
    face1 = face_analysis.get(webcam_img)[0]
    face2 = face_analysis.get(uploaded_img)[0]
    swapped_img = swapper.get(webcam_img, face1, face2, paste_back=True)

    # Encode the processed image back to base64
    _, img_encoded = cv2.imencode('.jpg', swapped_img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Return the processed image as a JSON response
    return jsonify({'processed_image': 'data:image/jpeg;base64,' + img_base64})

if __name__ == '__main__':
    app.run(debug=True)
