## Facial Expression Recognition App using CNN and OpenCV

This repository contains code for a Facial Expression Recognition (FER) app built using Convolutional Neural Networks (CNN) and OpenCV library. The app can detect facial expressions in real-time using a pre-trained model on the FER-2013 dataset along with the Haar Cascade frontal face classifier.

### Setup Instructions

Follow these steps to set up and use the FER app:

1. **Clone the Repository:**
   ```
   git clone https://github.com/NithinChowdaryRavuri/facial_expression_recognition.git
   cd fer-app
   ```

2. **Create and Activate Virtual Environment:**
   ```
   virtualenv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Model:**
   Download the pre-trained CNN model weights for FER-2013 dataset and place it in this directory. You can obtain the model from [here](https://drive.google.com/file/d/1OTAflsrsYzeH9fCkSdulJjcNitZlu_hY/view?usp=drive_link).

5. **Edit Main file:**
   Update line 6 & line 8 with corresponding Haar Cascade classifier and model address of your machine.

6. **Run the App:**
   ```
   python main.py
   ```

### Usage

Once the app is running, follow these steps to use it:

1. **Launch the App:**
   The app window will open displaying the webcam feed.

2. **Face Detection:**
   The app will automatically detect faces in the webcam feed using the Haar Cascade classifier.

3. **Facial Expression Recognition:**
   Once a face is detected, the app will recognize the facial expression in real-time using the pre-trained CNN model and display the corresponding label on the screen.

4. **Exit the App:**
   Press `q` key to exit the app.


### Training the Model

If you want to train the facial expression recognition model yourself or retrain it with custom data, follow these steps:

1. **Prepare Training Data:**
   Ensure you have training data prepared in the appropriate format. The training data should consist of labeled images of facial expressions. You can use the FER-2013 dataset or prepare your custom dataset.

2. **Modify `gen.py`:**
   Open the `gen.py` Python script and modify it according to your training data. Update the paths to your training and testing samples, and adjust any other parameters as needed.

3. **Run `gen.py`:**
   Execute the `gen.py` script to start training the model. This script will preprocess the data, train the CNN model, and save the trained weights.

4. **Evaluate Model Performance:**
   After training, evaluate the performance of the model on your testing data to ensure it achieves satisfactory accuracy and generalization.

5. **Integrate Trained Model:**
   Once you are satisfied with the trained model's performance, integrate it into the Facial Expression Recognition app by replacing the existing pre-trained model file with your newly trained weights.

### Note
Ensure you have sufficient computational resources for training the model, especially if working with large datasets or complex CNN architectures. Additionally, consider using techniques like data augmentation and transfer learning to improve model performance.

### License

MIT license.
