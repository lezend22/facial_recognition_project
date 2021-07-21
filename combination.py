import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('pythonProject/trainer/trainer.yml')
cascadePath = 'pythonProject/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0 #just init
names = ['seunghwan', 'mouse', 'cat']   #처음 등록했던 face_id순서대로 배열입력



model = load_model('pythonProject/trainer/VGG19-Face Mask Detection.h5')
model.summary()

# open webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
minW = 0.1 * webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1 * webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)


if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not status:
        print("Could not read frame")
        exit()

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(int(minW), int(minH))
    )


    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
            0] and 0 <= endY <= frame.shape[0]:

            face_region = frame[startY:endY, startX:endX]

            face_region1 = cv2.resize(face_region, (128, 128),interpolation=cv2.INTER_AREA)  # shape size(128,128) 바꿀려면 모델 제작시 바꿔야함

            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)

            print(prediction)
            if prediction[0][0].round() == 0:  # 마스크 미착용으로 판별되면,
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask ({:.2f}%)".format((1 - prediction[0][0]) * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                for (x, y, w, h) in faces:  #마스크 미착용시 얼굴인식 실시
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    id, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # id assigned

                    if confidence < 55:
                        id = names[id]
                    else:
                        id = "unknown"

                    confidence = "  {0}%".format(round(100 - confidence))

                    cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (0, 255, 0), 2)
                    cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (238, 75, 43), 1)

            elif prediction[0][0].round() == 1:  # 마스크 착용으로 판별되면
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask ({:.2f}%)".format(prediction[0][0] * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display output(웹캠으로 마스크 유무 식별)
    cv2.imshow("mask nomask classify", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()