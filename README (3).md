# Diabetic Retinopathy Detection
# Overview
This project aims to detect Diabetic Retinopathy from retinal images using deep learning techniques. Diabetic Retinopathy is a serious eye condition that can lead to blindness if not detected early. This model classifies the severity of Diabetic Retinopathy into five categories, enabling early diagnosis and treatment.

# Dataset
The dataset consists of high-resolution retinal images labeled into five classes:

* No DR: No Diabetic Retinopathy
- Mild: Mild Diabetic Retinopathy
- Moderate: Moderate Diabetic Retinopathy
- Severe: Severe Diabetic Retinopathy
- Proliferative DR: Proliferative Diabetic Retinopathy
The dataset can be found on Kaggle and should be downloaded  before training the model.

# Model Architecture
This project uses the VGG16 architecture with the last half of the layers unfrozen for fine-tuning. Despite its initial performance, the model's accuracy 70%. To improve performance, consider using alternative architectures such as:

ResNet50 or ResNet101: Known for deep learning capabilities with residual connections.
InceptionV3: Handles variations in image scales.
EfficientNet: Balances accuracy and efficiency with fewer parameters.

# Preprocessing
Images are resized to the input size required by the model.
Data augmentation is applied to increase model robustness, including rotations, flips, and brightness adjustments.
Training
Optimizer: Adam with a learning rate of 1e-4.
Loss Function: Categorical Crossentropy.
Metrics: Accuracy.
Batch Size: 32.
Results
The current model achieves an accuracy of 66% on the validation set.
Further improvements can be made by exploring different architectures or performing hyperparameter tuning.
# How to Use
# Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Keras
- OpenCV
- Matplotlib
- NumPy
# Installation
Clone the repository:

git clone https://github.com/yourusername/diabeticretinopathy.git
cd diabeticretinopathy
# Install the required packages:

pip install -r requirements.txt
Training the Model
Download and prepare the dataset as described above.
Run the training script:

python train.py
Model Inference
You can use the trained model to make predictions on new images:

python
Copy code
from tensorflow.keras.models import load_model
import cv2

# Load the model
model = load_model('diabetic_retinopathy_model.h5')

# Load and preprocess the image
img = cv2.imread('path_to_image.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = img.reshape(1, 224, 224, 3)

# Make prediction
prediction = model.predict(img)
predicted_class = prediction.argmax(axis=-1)

# Output the result
print(f'Predicted class: {predicted_class}')
Download the Model
You can download the trained model from the following link: Download Model

# Contributing
Contributions are welcome! Please create a pull request or raise an issue for any suggestions or improvements.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

