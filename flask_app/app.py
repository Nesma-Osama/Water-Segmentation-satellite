from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU
import tensorflow as tf
import numpy as np
import imageio.v3 as iio
import base64
import io
import os
from PIL import Image

# Flask setup
app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/'
)

# Constants
IMAGE_PATH = os.path.join(app.static_folder, 'images', '0.tif')
MASK_PATH = os.path.join(app.static_folder, 'images', '0.png')
MODEL_PATH= os.path.join(app.static_folder, 'model', 'model.h5')
REQUIRED_CHANNELS = 12
REQUIRED_DIM = (128, 128)

# Band names
bands = [
    "Coastal Aerosol", "Blue", "Green", "Red", "Near Infrared (NIR)",
    "Shortwave Infrared 1 (SWIR1)", "Shortwave Infrared 2 (SWIR2)",
    "QA Band (Quality Assessment)", "MERIT Digital Elevation Model (DEM)",
    "Copernicus DEM", "ESA World Cover Map", "Water Occurrence Probability"
]
####model
class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(DoubleConv, self).__init__(**kwargs)  # Pass kwargs to base Layer
        self.conv1 = Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv2 = Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu2(x)


model = load_model(MODEL_PATH, custom_objects={'DoubleConv': DoubleConv})
############################################################

# Convert channel to base64
def channel_to_base64(channel):
    normalized = (channel - channel.min()) / (np.ptp(channel) + 1e-5) * 255
    image = Image.fromarray(normalized.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


# Validate uploaded image
def validate_uploaded_image(image):
    if image.shape[-1] != REQUIRED_CHANNELS:
        return "Image must have 12 bands", 400
    if image.shape[:2] != REQUIRED_DIM:
        return "Image must be 128x128 pixels", 400
    return None

# segmentation
def segmentation(image):
    for channel in range(image.shape[-1]):
        image[:,:,channel]=(image[:,:,channel]-image[:,:,channel].min())/(image[:,:,channel].max()-image[:,:,channel].min()+ 1e-8)

    #batch size of 1
    image_expanded = np.expand_dims(image, axis=0)
    segmentation_result = model.predict(image_expanded)
    segmentation_result = np.squeeze(segmentation_result, axis=0)
    segmentation_result_binary = (segmentation_result > 0.5).astype(np.float32)
    segmentation_result_binary = segmentation_result_binary[:,:,0]  # Resize to original dimensions

    return segmentation_result_binary    



# Home route
@app.route('/', methods=['GET'])
def index():
    if not os.path.exists(IMAGE_PATH) or not os.path.exists(MASK_PATH):
        return render_template('error.html', message="Image or mask file not found.")

    image = iio.imread(IMAGE_PATH)
    mask = iio.imread(MASK_PATH)

    channels = [channel_to_base64(image[:, :, i]) for i in range(REQUIRED_CHANNELS)]
    mask_base64 = channel_to_base64(mask)

    return render_template('index.html', channels=channels, mask=mask_base64, bands=bands,REQUIRED_CHANNELS=REQUIRED_CHANNELS)


# Upload route
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file:
        return "No file uploaded", 400

    image = np.array(iio.imread(file.read()), dtype=np.float32)

    validation_error = validate_uploaded_image(image)
    if validation_error:
        return validation_error

    channels = [channel_to_base64(image[:, :, i]) for i in range(REQUIRED_CHANNELS)]
    mask=segmentation(image)
    if(mask is None):
        return "Segmentation failed", 500
    mask_base64 = channel_to_base64(mask)
    return render_template('segmentation.html', channels=channels,mask=mask_base64, bands=bands,REQUIRED_CHANNELS=REQUIRED_CHANNELS)


# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555, debug=True)
