import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r"C:\Users\DELL\Desktop\faiz\Species+Classifier\Species Classifier\model.h5") #Replace the Path              
# Define the categories
categories = ["cats", "dogs","elephants", "human", "Peacock", "pigs"]
# Define the live camera
cap = cv2.VideoCapture(0)
while True:
    # Capture a frame from the live camera
    ret, frame = cap.read()

    # Preprocess the image
    image = cv2.resize(frame, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Predict the class of the image
    prediction = model.predict(image)[0]
    class_idx = np.argmax(prediction)
    class_label = categories[class_idx]
    confidence = prediction[class_idx] * 100

    # Display the predicted class label and confidence
    cv2.putText(frame, f"{class_label} {confidence:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the live camera feed
    cv2.imshow("Live Camera", frame)

    # Wait for a key event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the live camera
cap.release()
cv2.destroyAllWindows()
