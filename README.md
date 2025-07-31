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

🎯 Pre-trained UNet
