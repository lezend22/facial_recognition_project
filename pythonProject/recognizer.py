import cv2
import numpy as np
from datetime import datetime
from pythonProject.backend.handler import insert_item_one, get_names

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer_update.yml')
cascadePath = 'haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0  # just init
#db에서 names 받아와야함

names = get_names()  # 처음 등록했던 face_id순서대로 배열

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

count = 0
trashCount = 0
todayVisited = []
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(int(minW), int(minH))
    )

    flag = False
    flagID = -1
    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])  # id assigned


        if confidence < 55:
            name = names[id]
            flag = True
            flagID = id
        else:
            name = "unknown"

        confidencePercentage = "  {0}%".format(round(100 - confidence))


        # confidence가 55이하를 충족하지못하여, 이전 루프에서 받았던 flagID과 다를때 trashCount만 커지고 count는 안커짐
        if flag == True and flagID == id:
            count += 1
            print("count", count)

            if count == 20 or id in todayVisited:
                cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (0, 128, 0), 2)
                cv2.putText(img, str(confidencePercentage), (x + 5, y + h - 5), font, 1, (0, 128, 0), 1)
                if id not in todayVisited:
                    todayVisited.append(id)
                    # confirmed. send id to db
                    time = datetime.now()
                    print(time)
                    # cursor = dbConnection()
                    #insert_item_one(id, time) # 풀면 진짜 db에 들어가요

                #init count, flags
                count = 0
                flagId = -1
                flag = False
                print("successfully sended")

            else:
                cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 0, 0), 2)
                cv2.putText(img, str(confidencePercentage), (x + 5, y + h - 5), font, 1, (255, 0, 0), 1)



    print("trash", trashCount)
    trashCount += 1
    if trashCount == 40:
        print("trash collector occurs")
        count = 0
        trashCount = 0
        flag = False
        flagId = -1

    cv2.imshow('camera', img)
    if cv2.waitKey(1) > 0: break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
