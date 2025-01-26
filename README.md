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
<p>Deep Learning Framework: TensorFlow, Keras</p>
<p>Model Type: Convolutional Neural Network (CNN)</p>
<p>Programming Language: Python</p>
<p>Image Processing: OpenCV</p>
<p>Deployment: Streamlit (for creating a simple web interface)</p>
<h3>Model Architecture Overview</h3>
<p>The model consists of a Convolutional Neural Network (CNN) architecture, which is well-suited for image classification tasks. The architecture includes multiple convolutional layers followed by fully connected layers to classify the image input into one of the predefined classes (e.g., "Not Distracted," "Talking to Passenger," "Using Phone," etc.).</p>

<h3>Layers in the Model:</h3>
<p>Convolutional Layers: These layers apply filters to the input images to extract features.
Max-Pooling Layers: Reduce the spatial dimensions of the image, making the model computationally efficient.
Fully Connected Layers: Used for decision-making based on the extracted features.
Output Layer: The final layer outputs the classification result, indicating the driver’s activity.</p>
<h3>Training Process</h3>
<p>The model is trained using a labeled dataset of images, where each image is labeled with the corresponding driver activity. We use a categorical cross-entropy loss function and Adam optimizer for training. The model is trained for multiple epochs to minimize the loss and improve accuracy.</p>

<h3>Dataset</h3>
<p></p>Source: The dataset used to train the model contains labeled images of drivers engaging in different activities while driving. Each image is labeled with a class indicating whether the driver is distracted or not, and if distracted, the type of distraction (e.g., using a phone, talking to passengers, etc.).</p>
<p>Data Augmentation: Techniques like rotation, flipping, and scaling were applied to augment the dataset and improve model robustness.</p>
<h3>Example Classes:</h3>
<p>Not Distracted: The driver is focused on driving.
Talking to Passenger: The driver is talking to someone in the vehicle.
Using Phone: The driver is using their mobile phone.
Eating: The driver is eating or drinking.</p>
<h3>How It Works</h3>
<p>Input: The model takes an image of the driver as input.</p>
<p>Preprocessing: The image is resized and normalized before being passed to the model.</p>
<p>Inference: The model predicts the class of the activity based on the input image.</p>
<p>Output: The result is displayed, showing whether the driver is distracted and if so, what activity they are engaged in.</p>

<h3>Results & Evaluation</h3>
<p>Accuracy: The model achieves an accuracy of 95% on the test dataset, showing strong performance in distinguishing between distracted and non-distracted driving.</p>
<p>Confusion Matrix: The confusion matrix indicates that the model has a low false positive and false negative rate, which is critical for real-time applications.</p>
<h3>Sample Output:</h3>
<p>Input: Image of a driver using a phone.</p>
<p>Output: "Driver is distracted (Using Phone)."</p>
<h3>Future Improvements</h3>
<p>Integration with Vehicle Systems: The model can be integrated with in-vehicle cameras for real-time monitoring.</p>
<p>Detection of More Activities: The model can be expanded to detect additional activities like reading or adjusting the rearview mirror.</p>
<p>Model Optimization: Further optimization of the model's performance for faster inference on lower-powered devices.</p>
<h3>Conclusion</h3>
<p>The Distracted Driver Detection Model is a powerful tool for improving road safety by automatically monitoring and detecting driver activities. By leveraging deep learning, the system can alert the driver or fleet manager about dangerous distractions, ultimately contributing to safer driving.</p>





