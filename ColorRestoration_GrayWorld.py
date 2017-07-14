import numpy as np
import cv2
from numpy import *
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

'''
Algoritmo GrayWorld assumption
En la primera parte, se aplica el algoritmo para obtener una imagen
mejorada de la imagen submarina. 
'''

#-----Lectura de Imagen-----------------------------------------------------
img = cv2.imread('MVI_0234_Cap1.png')
imgDehaze = cv2.imread('imagenRecuperadaDehaze.jpg')

#print img.dtype
#print img.shape

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow("img",img)

#-----Construccion de imagen de Salida-----------------------------------------
filas,columnas,canales = img.shape
outputImage = np.zeros((filas,columnas,canales))

#-----Calculo de Promedios-----------------------------------------------------
meanRed = np.mean(np.mean(img[:,:,0]))
meanGreen = np.mean(np.mean(img[:,:,1]))
meanBlue = np.mean(np.mean(img[:,:,2]))

#meanRed = np.mean(img[:,:,0])
#meanGreen = np.mean(img[:,:,1])
#meanBlue = np.mean(img[:,:,2])

#-----Calculo de Luminancia por c/canal-----------------------------------------------
lRed = meanRed/127
lGreen = meanGreen/127
lBlue = meanBlue/127

#print meanRed
#print meanGreen
#print meanBlue

#-----Balance de Blancos-----------------------------------------------------
outputImage[:,:,0] = img[:,:,0]*(1/lRed)
outputImage[:,:,1] = img[:,:,1]*(1/lGreen)
outputImage[:,:,2] = img[:,:,2]*(1/lBlue)

outputImageN = np.uint8(outputImage)

cv2.namedWindow('Salida',cv2.WINDOW_NORMAL)
cv2.imshow('Salida', outputImageN)

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-----Recuperacion del Color Rojo-----------------------------------------------------
'''
Algoritmo GrayWorld assumption
En la segunda parte, se usa la misma imagen de entrada para recuperar el color rojo
que no está presente en la imagen submarina. En esta parte se necesita aplicar otros
algoritmos de preprocesamiento (ej Dehazing)
'''
bDehaze,gDehaze,rDehaze = cv2.split(imgDehaze)
bOriginal,gOriginal,rOriginal = cv2.split(img)

#A partir de (avgRed+avgBlue+avgGreen)/3, se llega a:
avgGreen = np.mean(np.mean(imgDehaze[:,:,1]))
avgBlue = np.mean(np.mean(imgDehaze[:,:,0]))
print avgBlue
print avgGreen

MAX_avgBlue = np.max(bDehaze)
print MAX_avgBlue
min_avgBlue = np.min(bDehaze)
print min_avgBlue

MAX_avgGreen = np.max(gDehaze)
print MAX_avgGreen
min_avgGreen = np.min(gDehaze)
print min_avgGreen

nor_avgBlue = (avgBlue-min_avgBlue)/(MAX_avgBlue-min_avgBlue)
print nor_avgBlue
nor_avgGreen = (avgGreen-min_avgGreen)/(MAX_avgGreen-min_avgGreen)
print nor_avgGreen

nor_avgRed = 1.5-nor_avgBlue-nor_avgGreen
print nor_avgRed

avgR = np.mean(np.mean(img[:,:,2]))
print avgR

MAX_avgR = np.max(rOriginal)
print MAX_avgR
min_avgR = np.min(rOriginal)
print min_avgR

nor_avgR = (avgR-min_avgR)/(MAX_avgR-min_avgR)
print nor_avgR

delta = nor_avgRed/nor_avgR
print delta

nor_R = cv2.normalize(rOriginal,rOriginal,0,255,cv2.NORM_MINMAX)
#nor_R = normalize(img[:,:,2])
print nor_R
Rrecuperado = nor_R*delta
print Rrecuperado

Rrec8 = np.array(Rrecuperado,dtype=np.uint8)
imgequ = cv2.merge((bDehaze,gDehaze,Rrec8))


cv2.namedWindow('Rrecuperado',cv2.WINDOW_NORMAL)
cv2.imshow('Rrecuperado', Rrecuperado)
cv2.namedWindow('imgR',cv2.WINDOW_NORMAL)
cv2.imshow('imgR', imgequ)

color = ('b','g','r')
for i, col in enumerate(color):
	histcolorOriginal =  cv2.calcHist([imgequ],[i],None,[256],[0,256])
	cv2.normalize(histcolorOriginal,histcolorOriginal,8,cv2.NORM_MINMAX)
	plt.plot(histcolorOriginal,color=col)
plt.show()



cv2.waitKey()
cv2.destroyAllWindows()
