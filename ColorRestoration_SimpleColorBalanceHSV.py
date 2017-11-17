#!/usr/bin/env python
'''
Modulo 2 Toolbox
Color Restoration with Simplest Color Balance
Tesis Underwater Image Pre-processing
Armando Longart 10-10844
ajzlongart@gmail.com

Descripcion: Modulo implementado para mejorar el color de las imagenes
subacuaticas. Se basa en estirar el histograma (histogram stretching)
de la imagen haciendo que los colores de la imagen de salida este
mejorada. Se uso el modelo de color HSV para el algoritmo

Modulo implementado en Python

Variacion del Simple Color Balance (en RGB) de DavidYKay: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc 

'''
# Python 2/3 compatibility
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

#-----Funciones a Implementar-----------------------------------------------------
def apply_mask(matrix, mask, fill_value):
	'''
	Funcion encargada de "crear" la matriz de valores enmascarados. Estos valores son
	determinados a partir de los valores altos y bajos del la funcion apply_threshold
	(low_mask, high-mask como mask y de low_value, high_value como fill_value).

	El orden de ejecucion es Hue, Saturation, Value
	'''
	masked = np.ma.array(matrix,mask=mask,fill_value=fill_value)
	return masked.filled()


def apply_threshold(matrix, low_value, high_value):
	'''
	Esta funcion se encarga de crear una matriz booleana cuyos valores (True, False)
	dependeran de que los valores HSV de la imagen original sean mayores o menores que
	los valores umbrales maximos o minimos obtenidos en la funcion sColorBalance.

	El orden de ejecucion es Hue, Saturation, Value
	'''
	low_mask = matrix<low_value
	matrix = apply_mask(matrix,low_mask,low_value)

	high_mask = matrix>high_value
	matrix = apply_mask(matrix,high_mask,high_value)

	return matrix


def sColorBalance(img_hsv, porcentaje):
	'''
	Funcion encarganda de:
	Separar los canales HSV de la imagen (split)
	Ordenar los valores de pixeles y seleccionar los "cuantiles" de la matriz ordenada
	Obtener los valores max y min de la matriz ordenanda para cada canal HSV (a partir
		del porcentaje de saturacion)
	Saturar la imagen para los valores max y min de cada canal

	Todo esto con el fin de que los colores HSV ocupen el mayor rango posible [0,255]
	aplicando una transformacion afin a cada canal
	'''
	assert img_hsv.shape[2] == 3
	assert porcentaje > 0 and porcentaje < 100

	mitad_porcentaje = porcentaje/200.0
	canales = cv2.split(img_hsv)		#Separa los canales en HSV

	salida_canales = []
	for canal in canales:
		assert len(canal.shape) == 2
		# find the low and high precentile values (based on the input percentile)
		filas,columnas = canal.shape
		vec_tam = columnas*filas
		flat = canal.reshape(vec_tam)

		assert len(flat.shape) == 1

		flat = np.sort(flat)

		n_cols = flat.shape[0]

		#Seleccion de los valores minimos y maximos de cada canal HSV de la imagen. Seria el stretching
		bajo_val  = flat[math.floor(n_cols*mitad_porcentaje)]		#Calcula el valor bajo del arreglo ordenado de la matriz (img) de entrada para cada canal
		alto_val = flat[math.ceil(n_cols*(1-mitad_porcentaje))]		#Calcula el valor alto del arreglo ordenado de la matriz (img) de entrada para cada canal 			Alternativa: alto_val = flat[math.ceil(n_cols*(1-mitad_porcentaje)-1)]

		#Los valores alto y bajo para cada canal HSV. El orden de impresion es Hue, Saturation, Value
		print "Lowval: ", alto_val
		print "Highval: ", bajo_val

		# saturate below the low percentile and above the high percentile
		thresholded = apply_threshold(canal,bajo_val,alto_val)
		# scale the canal
		normalized = cv2.normalize(thresholded,thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
		salida_canales.append(normalized)
		norm_tile = np.tile(normalized[:,:,np.newaxis],(1,1,3))
	
		
	return cv2.merge(salida_canales)


if __name__ == '__main__':
	#Constuccion del parse y del argumento
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True, help = "Imagen de Entrada")
	args = vars(ap.parse_args())

	imgOriginal = cv2.imread(args["image"])

	#-------------------Creacion del archivo--------------------------
	f = open('img_txt.txt','a') #Tambien sirve open('img_txt.txt')  Archivo para colocar los resultados de los analisis cuantitativos. Sera append

	img_hsv = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV) 		#Conversion de HSV de la imagen original a HSV

	#-----Llamado a Funcion----------------------------------------------------
	imgRecuperada = sColorBalance(img_hsv, 1)	#Porcentaje de umbral inferior y superior respecto al histograma de entrada. Este porcentaje puede ser distinto para c/limite del histograma
	imgRecuperadaRGB = cv2.cvtColor(imgRecuperada,cv2.COLOR_HSV2BGR)	#Conversion de HSV de la imagen recuperada a RGB

	#-----Resultados----------------------------------------------------
	cv2.imshow("imgOriginal", imgOriginal)
	cv2.imshow("imgRecuperadaHSV", imgRecuperada)
	cv2.imshow("imgRecuperada", imgRecuperadaRGB)

	#-----Guardado de la imagen Recuperada-------------------------------------------
	cv2.imwrite('imagenRecuperadaCR_HSV_RGB.jpg',imgRecuperadaRGB)

	#-----Calculo de Histograma----------------------------------------------------
	'''
	Se calcula el histograma de la imagen Original en HSV para conocer el rango para aplicar
	el algoritmo
	'''
	hue,sat,val = cv2.split(img_hsv) 
	
	plt.subplot(311)                             #plot in the first cell
	plt.subplots_adjust(hspace=.5)
	plt.title("Hue Original")
	plt.ylabel('Numero de Pixeles')
	plt.hist(np.ndarray.flatten(hue), bins=128)
	plt.xlim([0,180])

	plt.subplot(312)                             #plot in the second cell
	plt.title("Saturation Original")
	plt.ylabel('Numero de Pixeles')
	plt.hist(np.ndarray.flatten(sat), bins=128)
	plt.xlim([0,256])

	plt.subplot(313)                             #plot in the third cell
	plt.title("Luminosity Value Original")
	plt.ylabel('Numero de Pixeles')
	plt.hist(np.ndarray.flatten(val), bins=128)
	plt.xlim([0,256])

	plt.show()

	'''
	Aqui se calcula el histograma de la imagen Original y la Recuperada en RGB
	'''
	color = ('b','g','r')
	for i, col in enumerate(color):
	   histcolorOriginal =  cv2.calcHist([imgOriginal],[i],None,[256],[0,256])
	   histcolorRecuperadaRGB =  cv2.calcHist([imgRecuperadaRGB],[i],None,[256],[0,256])

	   plt.subplot(211), plt.plot(histcolorOriginal,color=col)
	   plt.title('Histograma Original')
	   plt.ylabel('Numero de Pixeles')
	   plt.xlim([0,256])

	   plt.subplot(212), plt.plot(histcolorRecuperadaRGB,color=col)
	   plt.title('Histograma Recuperada')
	   plt.ylabel('Numero de Pixeles')
	   plt.xlabel('Bins')
	   plt.xlim([0,256])

	plt.show()

	#-----Analisis Cuantitativo de la Imagen------------------------------------------------
	'''
	Se realizan 3 tipos de analisis de la imagen resultante:
	Espectro Frecuencial
	Entropia
	Deteccion de features
	Con la finalidad de saber cual es el algoritmo que mejor funciona para las imagenes
	submarinas
	'''

	#Espectro Frecuencial
	IMG = cv2.imread(args["image"],0)
	IMGRec = cv2.imread('imagenRecuperadaCR_HSV_RGB.jpg',0)
#	IMG = cv2.resize(IMG, (1024, 768))
#	IMGRec = cv2.resize(IMGRec, (1024, 768))
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

	plt.subplot(311),plt.imshow(img32,cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(312),plt.imshow(20*np.log(mod_fourier32))
	plt.title('Magnitude Spectrum Original'), plt.xticks([]), plt.yticks([])
	plt.subplot(313),plt.imshow(20*np.log(mod_fourierRec32))
	plt.title('Magnitude Spectrum Rec'), plt.xticks([]), plt.yticks([])
	plt.show()


	#Entropia de la imagen a partir del histograma de grises de la iamgen
	histogramIMG = cv2.calcHist([IMG],[0],None,[256],[0,256])
	cv2.normalize(histogramIMG,histogramIMG,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	histIMG = histogramIMG.sum()
	probIMG = [float(h)/histIMG for h in histogramIMG]
	entropyIMG = -np.sum([p*np.log2(p) for p in probIMG if p !=0])
	print entropyIMG

	histogramIMGRec = cv2.calcHist([IMGRec],[0],None,[256],[0,256])
	cv2.normalize(histogramIMGRec,histogramIMGRec,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	histIMGRec = histogramIMGRec.sum()
	probIMGRec = [float(h)/histIMGRec for h in histogramIMGRec]
	entropyIMGRec = -np.sum([p*np.log2(p) for p in probIMGRec if p !=0])
	print entropyIMGRec

	#-----Escritura del archivo con los resultados----------------------------------------------
	#Con write()
	f.write('%s \t %d \t %d \t %f \t %f \t %f \t %f \t HSVStretch \n' %(args["image"], row, col, iqm32, iqmRec32, entropyIMG, entropyIMGRec))
	f.close()

	cv2.waitKey(0)
	cv2.destroyAllWindows()
