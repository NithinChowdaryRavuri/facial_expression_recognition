import tensorflow as tf
import cv2
import numpy as np

#use the haarcascade_frontalface_default.xml file to detect the face
face_classifier = cv2.CascadeClassifier(r'/Users/nithinchowdaryravuri/Desktop/deep_learning/haarcascade_frontalface_default.xml')
#load the model
classifier = tf.keras.models.load_model(r'/Users/nithinchowdaryravuri/Desktop/deep_learning/model.h5')

#list of emotions
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

#capture the video from the webcam
cap = cv2.VideoCapture(1)

while True:
    #read the frame
    _, frame = cap.read()
    labels = []
    #convert the frame to gray scale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detect the face in the frame
    faces = face_classifier.detectMultiScale(gray)

    #draw a rectangle around the face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(64,64),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            #preprocess the image
            roi = roi_gray.astype('float')/255.0
            roi = np.expand_dims(roi,axis=0)
            roi = roi.reshape((-1, 64, 64, 1))
            #make a prediction
            prediction = classifier.predict(roi)
            #find the label of the emotion
            label=emotion_labels[prediction.argmax()]
            #add the label to the list
            label_position = (x,y)
            #display the label on the screen
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    #if q is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#release the video capture
cap.release()
#destroy all windows
cv2.destroyAllWindows()