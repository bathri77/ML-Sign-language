import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import threading
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)


# Function for speech synthesis
def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
        print(text)
    except Exception as e:
        print(text)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    detector = HandDetector(maxHands=2)  # Set maxHands to 2 for multi-hand detection
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
    offset = 20
    imgSize = 300
    counter = 0

    labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

    while True:
        try:
            success, img = cap.read()
            if not success:
                print("Error: Unable to capture frame.")
                continue

            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                for hand in hands:  # Loop through detected hands
                    x, y, w, h = hand['bbox']

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                    imgCropShape = imgCrop.shape

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap: wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)

                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap: hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)

                    # Get the predicted label
                    predicted_label = labels[index]

                    # Display the predicted label on the image
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50),
                                  (0, 255, 0),
                                  cv2.FILLED)
                    cv2.putText(imgOutput, predicted_label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x+ w + offset, y + h + offset), (0, 255, 0), 4)

                    # Convert the predicted label to speech in a separate thread
                    threading.Thread(target=speak, args=(predicted_label,)).start()

                    #cv2.imshow('ImageCrop', imgCrop)
                    cv2.imshow('ImageWhite', imgWhite)

            cv2.imshow('Image', imgOutput)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error occurred: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
