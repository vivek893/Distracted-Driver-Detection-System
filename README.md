# Distracted-Driver-Detection-System
![Model Architecture](imgs/model%20Architecture%20pic.png)
<h3>Project Overview</h3>
<p>This project implements a Distracted Driver Detection Model that uses deep learning to classify driver activities. The model detects whether the driver is distracted (e.g., using a phone, operating the radio, talking to a passenger, etc.) based on images captured from a camera. The aim is to improve road safety by monitoring drivers' behavior in real-time.</p>

<p>The model is based on Convolutional Neural Networks (CNNs) and uses TensorFlow for training and deployment. It is designed to identify and classify multiple driver activities, providing a valuable tool for automated monitoring systems.</p>

<h3>Key Features</h3>
<p></p>Real-Time Monitoring: The model processes images of drivers in real time to classify their behavior.
Multi-Class Classification: Detects multiple activities such as using a phone, talking to passengers, eating, etc.
Accurate Detection: The model uses advanced CNN techniques to achieve high accuracy in classifying driver activity.
Deployable in Vehicle Systems: Can be integrated into vehicle monitoring systems for continuous safety monitoring.</p>
<h3>Technologies Used</h3>
Deep Learning Framework: TensorFlow, Keras
Model Type: Convolutional Neural Network (CNN)
Programming Language: Python
Image Processing: OpenCV
Deployment: Streamlit (for creating a simple web interface)
Model Architecture
Overview
The model consists of a Convolutional Neural Network (CNN) architecture, which is well-suited for image classification tasks. The architecture includes multiple convolutional layers followed by fully connected layers to classify the image input into one of the predefined classes (e.g., "Not Distracted," "Talking to Passenger," "Using Phone," etc.).

Layers in the Model:
Convolutional Layers: These layers apply filters to the input images to extract features.
Max-Pooling Layers: Reduce the spatial dimensions of the image, making the model computationally efficient.
Fully Connected Layers: Used for decision-making based on the extracted features.
Output Layer: The final layer outputs the classification result, indicating the driver’s activity.
Training Process
The model is trained using a labeled dataset of images, where each image is labeled with the corresponding driver activity. We use a categorical cross-entropy loss function and Adam optimizer for training. The model is trained for multiple epochs to minimize the loss and improve accuracy.

Dataset
Source: The dataset used to train the model contains labeled images of drivers engaging in different activities while driving. Each image is labeled with a class indicating whether the driver is distracted or not, and if distracted, the type of distraction (e.g., using a phone, talking to passengers, etc.).
Data Augmentation: Techniques like rotation, flipping, and scaling were applied to augment the dataset and improve model robustness.
Example Classes:
Not Distracted: The driver is focused on driving.
Talking to Passenger: The driver is talking to someone in the vehicle.
Using Phone: The driver is using their mobile phone.
Eating: The driver is eating or drinking.
How It Works
Input: The model takes an image of the driver as input.
Preprocessing: The image is resized and normalized before being passed to the model.
Inference: The model predicts the class of the activity based on the input image.
Output: The result is displayed, showing whether the driver is distracted and if so, what activity they are engaged in.
Usage
1. Clone the Repository
To use the model, clone the repository to your local machine:

bash
Copy
Edit
git clone https://github.com/yourusername/distracted-driver-detection.git
cd distracted-driver-detection
2. Install Dependencies
Install the required libraries using pip:
pip install -r requirements.txt
3. Run the App
To test the model using a sample image, run the following command:

bash
Copy
Edit
streamlit run app.py
This will launch a Streamlit app in your web browser where you can upload images of drivers to detect if they are distracted.

4. Use the Model for Inference
For real-time monitoring, the model can be integrated with a camera system, where it continuously captures images of the driver and classifies their activity.
The app can also be customized to work with video streams for ongoing driver monitoring.
Results & Evaluation
Accuracy: The model achieves an accuracy of 95% on the test dataset, showing strong performance in distinguishing between distracted and non-distracted driving.
Confusion Matrix: The confusion matrix indicates that the model has a low false positive and false negative rate, which is critical for real-time applications.
Sample Output:
Input: Image of a driver using a phone.
Output: "Driver is distracted (Using Phone)."
Future Improvements
Integration with Vehicle Systems: The model can be integrated with in-vehicle cameras for real-time monitoring.
Detection of More Activities: The model can be expanded to detect additional activities like reading or adjusting the rearview mirror.
Model Optimization: Further optimization of the model's performance for faster inference on lower-powered devices.
Conclusion
The Distracted Driver Detection Model is a powerful tool for improving road safety by automatically monitoring and detecting driver activities. By leveraging deep learning, the system can alert the driver or fleet manager about dangerous distractions, ultimately contributing to safer driving.





