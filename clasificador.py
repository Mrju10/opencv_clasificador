#instala las librerias de cv2 con el comando pip si deseas 
import cv2
import os

imagesPath = 'D:/test/'#Aqui colocas la ruta donde estan las fotos
imagesPathList = os.listdir(imagesPath)
#ubica la carpeta en el lugar donde te encuentra
if not os.path.exists('dataset'):
    print('Carpeta creada: dataset')
    os.makedirs('dataset')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0
for imageName in imagesPathList:
    image = cv2.imread(imagesPath+'/'+imageName)
    imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)

    faces = faceClassif.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
            #rostro = imageAux[y:y+h,x:x+w]
            rostro = imageAux
            #rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
            #cv2.imshow('rostro',rostro)
            #cv2.waitKey(0)
            cv2.imwrite('Rostros encontrados/rostro_{}.jpeg'.format(count),rostro)
            count = count +1

cv2.destroyAllWindows()