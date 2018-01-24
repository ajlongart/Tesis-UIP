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
	ap.add_argument("-j", "--image2", required = True, help = "Imagen de Entrada Mejorada")
	args = vars(ap.parse_args())


	#Se usa el formato double para el algoritmo.
	img = double(cv2.imread(args["image"]))/255 #/255
	#Usado para calcular el histograma y la conversion al canal YCrCb. La imagen 
	#para ambos casos debe ser o int 8bits, o int 16bits o float 32bits: cv2.cvtColor y calcHist
	imgOriginal = cv2.imread(args["image"])
	##Para reduccion, se usa Area. Para amplicacion, (Bi)Cubica INTER_CUBIC
	img = cv2.resize(img,None, fx=0.8,fy=0.8,interpolation=cv2.INTER_AREA)

	#-------------------Creacion del archivo--------------------------
	f = open('img_txtFourier.txt','a') #Tambien sirve open('img_txt.txt') Archivo para colocar los resultados de los analisis cuantitativos. Sera append	

	#Espectro Frecuencial
	IMG = cv2.imread(args["image"],0)
	IMGRec = cv2.imread(args["image2"],0)
#	IMG = cv2.resize(IMG,None, fx=0.8,fy=0.8,interpolation=cv2.INTER_AREA)
#	IMGRec = cv2.resize(IMGRec,None, fx=0.8,fy=0.8,interpolation=cv2.INTER_AREA)
	img32 = np.float32(IMG)
	imgRec32 = np.float32(IMGRec)

	row,col = np.shape(img32)

	fourier32 = np.fft.fft2(img32)/float(row*col)
	fourierShift32 = np.fft.fftshift(fourier32)
	mod_fourier32 = np.abs(fourierShift32)

	max_mod_fourier32 = np.max(mod_fourier32)
	thresh32 = max_mod_fourier32/1000
	thresh_fourier32 = mod_fourier32[(mod_fourier32>thresh32)]	#*mod_fourier
	tam_thresh_fourier32 = np.size(thresh_fourier32)

	iqm32 = tam_thresh_fourier32/(float(row*col))

	fourierRec32 = np.fft.fft2(imgRec32)/float(row*col)
	fourierShiftRec32 = np.fft.fftshift(fourierRec32)
	mod_fourierRec32 = np.abs(fourierShiftRec32)

	max_mod_fourierRec32 = np.max(mod_fourierRec32)
	threshRec32 = max_mod_fourierRec32/1000
	thresh_fourierRec32 = mod_fourierRec32[(mod_fourierRec32>threshRec32)]	#*mod_fourier
	tam_thresh_fourierRec32 = np.size(thresh_fourierRec32)

	iqmRec32 = tam_thresh_fourierRec32/(float(row*col))

	print iqm32
	print iqmRec32

#	plt.subplot(311),plt.imshow(img32,cmap = 'gray')
#	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#	plt.subplot(312),plt.imshow(20*np.log(mod_fourier32))
#	plt.title('Magnitude Spectrum Original'), plt.xticks([]), plt.yticks([])
#	plt.subplot(313),plt.imshow(20*np.log(mod_fourierRec32))
#	plt.title('Magnitude Spectrum Rec'), plt.xticks([]), plt.yticks([])
#	plt.show()

	#-----Escritura del archivo con los resultados----------------------------------------------
	#Con write()
	f.write('%s \t %d \t %d \t %f \t %f \t DehazingGWa \n' %(args["image"], row, col, iqm32, iqmRec32)
	f.write('%s \t %d \t %d \t %f \t %f \t DehazingGWa \n' %(args["image2"], row, col, iqm32, iqmRec32))
	f.close()

	cv2.waitKey()
	cv2.destroyAllWindows()
