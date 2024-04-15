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

### License

MIT license.
