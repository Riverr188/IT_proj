print("Flower Detection")
import requests
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time
import  sys
from  Adafruit_IO import  MQTTClient


AIO_USERNAME = "Riverr"
AIO_KEY = "aio_JzrD9267kXZBQAGLSgpEmDBbibdr"


client = MQTTClient(AIO_USERNAME , AIO_KEY)
client.connect()


# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

while True:
    try:
        image_url = input("Enter image URL: ")
        # Fetch the image from the URL
        response = requests.get(image_url)
        image_bytes = response.content
        # Convert the image bytes to a numpy array
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a windoư
        cv2.imshow("Linked Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        confidence_percent = str(np.round(confidence_score * 100))[:-2]

        # Print prediction and confidence score
        print("Predicted Class:", class_name)
        value = f"Predicted Class: {class_name}"
        print("Class:", class_name, end="")
        print("Confidence Score:", confidence_percent, "%")
        time.sleep(2)
        client.publish("ai", value)
        client.publish("conf", confidence_percent)
        
    except Exception as e:
        print(f"Error: {e}")

        time.sleep(2)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break
