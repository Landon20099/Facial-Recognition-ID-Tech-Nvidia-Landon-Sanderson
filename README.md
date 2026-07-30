Facial Emotion Detector



Hello my name is Landon Sanderson, I used my Jetson Orin Nano to create an emotion detector based on the person's facial expressions.


The Algorithm



The facial emotion recognition algorithm begins by continuously capturing frames from a webcam and applying a face detection model to locate and isolate the user's face from the background before any classification is performed. Each detected face is cropped, resized to 224 x 224 pixels, then converted into a numerical tensor, and normalized using the same preprocessing pipeline that was applied during the training to ensure consistent input regardless of lighting conditions, camera resolution, or face position. The preprocessed image is then passed into a MobileNetv3 convolution neural network (CNN) that was trained using transfer learning on a labeled dataset containing six emotion classes: happy, sad, anger, fear, disgust and pain. During training the network learned hierarchical facial features by repeatedly comparing its predictions to the correct labels. Calculating the classification error with a loss function, and updating millions of trainable parameters through backpropagation and gradient descent over multiple epochs. As a result, the CNN automatically recognizes discrimination facial characteristics such as eyebrow position, eye shape, mouth curvature, facial wrinkles, and overall facial geometry without requiring manually engineered features. During inference, the trained model extracts these learned features from each webcam frame, computes a probability score for every emotion class using its final classification layer, and selects the emotion with the highest confidence as the prediction. The application then overlays a bouncing box around the detected face and displays both the predicted emotion and its confidence score on the live video feed allowing users to observe the classification results in real time. By leveraging the NVIDIA Jetson Orin allowing the users to observe the classification results in real time. By leveraging the NVIDIA Jetson Orin Nano’s GPU acceleration, the system performs fast, low latency inference for embedded artificial intelligence applications that require accurate, real time facial emotion recognition.
Running this project
Add steps for running this project.
Make sure to include any required libraries that need to be installed for your project to run.





(https://drive.google.com/file/d/1alej2PpzzOcS57DREsBpXF_iubXsUymf/view?usp=sharing)



