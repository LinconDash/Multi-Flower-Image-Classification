# 🌸 Multi-Flower Image Classification 🌸

## Cloud Deployment : 
- https://flower-classification-app-269991126018.us-central1.run.app/

## Overview
The Multi-Flower Image Classification project is a deep learning-based image classification model designed to accurately identify and classify images of **17 different species of flowers**. It leverages **transfer learning** using the **VGG16** convolutional neural network (CNN) from Keras to achieve high accuracy with relatively low computational costs.

This project is ideal for **botanists, researchers, gardening enthusiasts, and AI practitioners** who want to classify flower species automatically based on image inputs. The model has been trained on a **diverse dataset** of flower images, making it robust against variations in lighting, angles, and backgrounds.

## Objectives
The main objectives of this project are:
-  Develop a robust classifier that can distinguish between 17 different flower species.
-  Leverage transfer learning to improve accuracy with minimal training data.
-  Optimize the model for real-world usability with preprocessing techniques.
-  Provide an easy-to-use interface (Flask/FastAPI) for end users.
-  Make the model lightweight so it can be deployed on cloud platforms or edge devices.

## Why Use VGG16?
VGG16 is a **pretrained deep learning model** known for its simple yet powerful architecture, making it an excellent choice for transfer learning. Key reasons for using VGG16:
-  **Pretrained on ImageNet**, which includes a vast number of objects and textures.
-  **Deep feature extraction capability** allows for excellent generalization.
-  **Lightweight** compared to modern CNNs, making it easier to deploy.
-  Works well with flower images, which often have **complex textures** and **color patterns**.

In this project, we remove the fully connected layers of VGG16 and replace them with a custom classification head, fine-tuning it for flower species classification.

# Flower Categories
This model is trained to classify the following 17 flower species:

**Bluebell,
Buttercup,
Colts-foot, 
Cowslip,
Crocus,
Daffodil,
Daisy,
Dandelion,
Fritillary,
Iris,
Lily-of-the-Valley / Lilly-Valley,
Pansy,
Snowdrop,
Sunflower,
Tigerlily,
Tulip, and Windflower**

Each of these flowers exhibits distinct petal arrangements, colors, and textures, making classification a challenging yet fascinating task for deep learning models.

## Project Workflow
The project follows a structured pipeline for training and inference:

**1. Dataset Preparation :**

- Images are collected and organized into train, validation directories.
- Data is prepared for the model using a custom data generator.
- Data augmentation techniques such as resizing, flipping, bluring, normalize, etc.  are applied to improve model generalization using albumentations module.
- Customized exception handling and logging is also implemented.

**2. Model Selection & Training :**

- VGG16 is used as a feature extractor.
- The fully connected layers are replaced with a custom classifier with a Softmax activation.
- The model is trained using Adam optimizer with **sparse categorical cross-entropy** loss.
- After the model is trained it is stored inside the `models/` directory

**3. Inference & Deployment :**

- The trained model is saved in .h5 format for easy reusability.
- A `app.py` script allows users to classify new images.
- A Flask web app is optionally built to enable an interactive UI.
- Along with the personalized response we have **Voice Assistant**
- Then it is containerized using the Dockerfile and Docker.
- The container is hosted in the **Cloud Run** of google cloud platform (GCP).

## Execution Guide
This guide provides a step-by-step walkthrough of setting up, training, and deploying the Multi-Flower Image Classification project. It includes cloning the repository, setting up a virtual environment, training the model, making predictions, and containerizing the project using Docker.

**1. Clone the Repository :**
First, clone the project from GitHub:
```
git clone https://github.com/your-username/multi-flower-classification.git
cd multi-flower-classification
```
**2. Set Up a Virtual Environment :**
It is recommended to use a virtual environment to manage dependencies.

- For Windows (cmd or PowerShell)
```
python -m venv venv
venv\Scripts\activate
```

- For macOS/Linux (Terminal)
```
python3 -m venv venv
source venv/bin/activate
```

Now, your terminal should show (venv), indicating the environment is active.

**3. Install Dependencies :**
Run the following command to install all required Python libraries:
```
pip install -r requirements.txt   #(or pip install .)
```
Make sure important libraries like TensorFlow, Keras, Albumantations, Matplotlib, NumPy, and Flask are installed for model training and deployment.

**4. Train the Model :**
Run the training script to start training the VGG16-based CNN model, but make sure your device has a **GPUs** for smoother training experience:
```
python -m src.pipeline.training.py
```
  - The trained model will be saved inside the `models/` directory.
  - Training progress, accuracy, and loss metrics will be displayed on the console.
  - Modify `src/pipeline/traininig.py` to adjust epochs, batch size, and learning rate if needed.

**5. Running Flask Web-App :**
Start the Flask application:
```
python app.py
```
- Then open your browser and visit: http://localhost:8080
- Upload an image, and the app will classify the flower.

**6. Containerize the Project with Docker :**
To make deployment easier, we will containerize the application

Build the Docker Image
Run the following command to build the Docker image:
```
docker build -t flower-classifier .
```

Run the Container
To run the containerized app:
```
docker run -p 5000:5000 flower-classifier
```
Now, open your browser and go to: http://localhost:8080

## Next Steps : 

- Add more data about other flower species for educational purposes
- Convert the model for mobile deployment using TensorFlow Lite
- Enhance the model using ResNet/EfficientNet for better accuracy
