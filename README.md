# 🌊 WaterLens – Detect Water from Space

WaterLens is an AI-powered web application designed to detect and highlight water bodies from satellite imagery.

## 📂 About the Data

Each input image is a multi-spectral image with dimensions 128x128x12, where the 12 channels represent:

1- Coastal Aerosol

2- Blue

3- Green

4- Red

5- Near Infrared (NIR)

6- Shortwave Infrared 1 (SWIR1)

7- Shortwave Infrared 2 (SWIR2)

8- QA Band (Quality Assessment)

9- MERIT Digital Elevation Model (DEM)

10- Copernicus DEM

11- ESA World Cover Map

12- Water Occurrence Probability

## 🧪 Models Used
- Custom UNet (From Scratch)
  Built manually using Keras 

  Tuned for performance on 128x128x12 multi-channel images.

- Pre-trained UNet

  - Keras: Used ResNet101 as the encoder backbone with the segmentation_models library.
  - PyTorch: Implemented using ResNet34 as the encoder via the segmentation_models_pytorch library.
 
## 🚀 How to Run the Flask App

1- Generate the Trained Model

  Open and run ```Water_Segmentation.ipynb``` to train the model. Once training is complete, save the   model as ```model.h5```.

2- Place the Model File

  Move the saved model.h5 file to the following path in your project directory:
```flask_app/static/model/model.h5```

3- Run the App Using Docker
  
  Pull the pre-built Docker image from Docker Hub:
  ```docker pull nesmaosama/segmentation-app```
  
  Then run the container:
  ```docker run -p 5555:5555 nesmaosama/segmentation-app```

  This will start the Flask server on http://localhost:5555
  
