import cv2
import numpy as np
from PIL import Image
import os


detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


def getImagesAndLabel_update(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #listdir : 해당 디렉토리 내 파일 리스트
    #path + file Name : 경로 list 만들기

    faceSamples = []
    ids = []
    for imagePath in imagePaths: #각 파일마다
        print(imagePath)
        #흑백 변환
        PIL_img = Image.open(imagePath).convert('L') #L : 8 bit pixel, bw
        print(PIL_img)
        img_numpy = np.array(PIL_img, 'uint8')

        #user id
        id = int(os.path.split(imagePath)[-1].split(".")[1])#마지막 index : -1
        print(id)
        #얼굴 샘플
        faces = detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples, ids

print('\n [INFO] Training faces. It will take a few seconds. Wait ...')

if __name__ == '__main__':
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    face_name = input("enter the user name ==> ") #일단 임시로 입력해서 찾아보자, 나중엔 main에서 face_name직접 받아올거
    path = 'dataset/'+ face_name
    faces, ids = getImagesAndLabel_update(path)
    recognizer.update(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')

