#!/usr/bin/env python
'''
Modulo 2 Toolbox
Analisis Cuantitativo de la Imagen
Tesis Underwater Image Pre-processing
Armando Longart 10-10844
ajzlongart@gmail.com

#-----Analisis Cuantitativo de la Imagen------------------------------------------------
Se realiza analisis de Entropia de las imagenes originales y resultante:
Con la finalidad de saber cual es el algoritmo que mejor funciona para las imagenes
submarinas

Modulo implementado en Python
'''
# Python 2/3 compatibility
import numpy as np
import cv2
from numpy import *
from matplotlib import pyplot as plt
import argparse
import os


if __name__ == '__main__':
	#-----Lectura de Imagen-----------------------------------------------------
	#Constuccion del parse y del argumento
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True, help = "Imagen de Entrada")
#	ap.add_argument("-j", "--image2", required = True, help = "Imagen de Entrada Mejorada")
	args = vars(ap.parse_args())


	#Se usa el formato double para el algoritmo.
	img = double(cv2.imread(args["image"]))/255 #/255
	#Usado para calcular el histograma y la conversion al canal YCrCb. La imagen 
	#para ambos casos debe ser o int 8bits, o int 16bits o float 32bits: cv2.cvtColor y calcHist
	imgOriginal = cv2.imread(args["image"])
	##Para reduccion, se usa Area. Para amplicacion, (Bi)Cubica INTER_CUBIC
	img = cv2.resize(img,None, fx=0.8,fy=0.8,interpolation=cv2.INTER_AREA)

	#-------------------Creacion del archivo--------------------------
	f = open('img_txtEntropia.txt','a') #Tambien sirve open('img_txt.txt') Archivo para colocar los resultados de los analisis cuantitativos. Sera append	

	IMG = cv2.imread(args["image"],0)
#	IMGRec = cv2.imread(args["image2"],0)
#	IMG = cv2.resize(IMG,None, fx=0.8,fy=0.8,interpolation=cv2.INTER_AREA)
#	IMGRec = cv2.resize(IMGRec,None, fx=0.8,fy=0.8,interpolation=cv2.INTER_AREA)
	img32 = np.float32(IMG)
#	imgRec32 = np.float32(IMGRec)

	row,col = np.shape(img32)

	#Entropia de la imagen a partir del histograma de grises de la iamgen
	histogramIMG = cv2.calcHist([IMG],[0],None,[256],[0,256])
	cv2.normalize(histogramIMG,histogramIMG,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	histIMG = histogramIMG.sum()
	probIMG = [float(h)/histIMG for h in histogramIMG]
	entropyIMG = -np.sum([p*np.log2(p) for p in probIMG if p !=0])
	print entropyIMG

#	histogramIMGRec = cv2.calcHist([IMGRec],[0],None,[256],[0,256])
#	cv2.normalize(histogramIMGRec,histogramIMGRec,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#	histIMGRec = histogramIMGRec.sum()
#	probIMGRec = [float(h)/histIMGRec for h in histogramIMGRec]
#	entropyIMGRec = -np.sum([p*np.log2(p) for p in probIMGRec if p !=0])
#	print entropyIMGRec

	#-----Escritura del archivo con los resultados----------------------------------------------
	#Con write()
	f.write('%s \t %d \t %d \t %f \n' %(args["image"], row, col, entropyIMG))
#	f.write('%s \t %d \t %d \t %f \t %f \n' %(args["image2"], row, col, entropyIMG, entropyIMGRec))
	f.close()

	cv2.waitKey()
	cv2.destroyAllWindows()
