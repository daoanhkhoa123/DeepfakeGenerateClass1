import os
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import insightface
from io import BytesIO
import cv2

app = Flask(__name__)

# Initialize InsightFace models
face_analysis = insightface.app.FaceAnalysis(name='buffalo_l')
face_analysis.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

def process_image(image_data, uploaded_face_data=None):
    # Convert base64 string to image
    image_data = base64.b64decode(image_data.split(',')[1])  # Remove the data URL part
    img1 = np.array(Image.open(BytesIO(image_data)))

    if uploaded_face_data:
        # If an uploaded face image is provided
        uploaded_face_data = base64.b64decode(uploaded_face_data.split(',')[1])
        img2 = np.array(Image.open(BytesIO(uploaded_face_data)))

        # Swap faces
        img1_ = swapper.get(img1, face_analysis.get(img1)[0], face_analysis.get(img2)[0], paste_back=True)
    else:
        # If no uploaded face is provided, just return the original webcam frame
        img1_ = img1

    # Convert the processed image back to base64
    _, buffer = cv2.imencode('.jpg', img1_)
    processed_image = base64.b64encode(buffer).decode('utf-8')

    return processed_image

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()

    webcam_frame = data['image']  # Webcam frame as base64 string
    uploaded_face = data.get('uploaded_face')  # Uploaded face as base64 string (might be None)

    # Process the image by swapping faces or returning the original webcam frame
    processed_image = process_image(webcam_frame, uploaded_face)

    return jsonify({
        'processed_image': 'data:image/jpeg;base64,' + processed_image
    })

if __name__ == "__main__":
    app.run(port=8012, debug=True)
