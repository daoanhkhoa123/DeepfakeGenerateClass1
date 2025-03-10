import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to process webcam frame
@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the image data from the request
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 string
    img_data = base64.b64decode(image_data.split(',')[1])

    # Convert to a numpy array for OpenCV processing
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Process the image (Example: Convert to grayscale)
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encode the processed image back to base64
    _, img_encoded = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Return the processed image as a JSON response
    return jsonify({'processed_image': 'data:image/jpeg;base64,' + img_base64})

if __name__ == '__main__':
    app.run(debug=True)
