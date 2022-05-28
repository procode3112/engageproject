import cv2
import numpy as np
import face_recognition

imgelon= face_recognition.load_image_file('ImagesBasics/download.jpg')
imgelon= cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgtest= face_recognition.load_image_file('ImagesBasics/download (1).jpg')
imgtest= cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
faceLoc= face_recognition.face_locations(imgelon)[0]
encodeElon=face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),1)
faceLocTest= face_recognition.face_locations(imgtest)[0]
encodeTest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),1)

result=face_recognition.compare_faces([encodeElon],encodeTest)
facdis=face_recognition.face_distance([encodeElon],encodeTest)
cv2.putText(imgtest,f'{result} {round(facdis[0],2)}',(25,25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
print(result,facdis)
cv2.imshow('Elon Musk',imgelon)
cv2.imshow('Elon Musk',imgtest)
cv2.waitKey(0)