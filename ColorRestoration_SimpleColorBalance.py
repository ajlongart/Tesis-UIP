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
mejorada. Se uso el modelo de color RGB para el algoritmo

Modulo implementado en Python
'''
# Python 2/3 compatibility
import cv2
import math
import numpy as np
import sys
from matplotlib import pyplot as plt

#-----Funciones a Implementar-----------------------------------------------------
def apply_mask(matrix, mask, fill_value):
	'''
	Funcion encargada de "crear" la matriz de valores enmascarados. Estos valores son
	determinados a partir de los valores altos y bajos del la funcion apply_threshold
	(low_mask, high-mask como mask y de low_value, high_value como fill_value).

	El orden de ejecucion es Blue, Green, Red
	'''
	masked = np.ma.array(matrix,mask=mask,fill_value=fill_value)
#	cv2.imshow("Masked", masked)
	return masked.filled()


def apply_threshold(matrix, low_value, high_value):
	'''
	Esta funcion se encarga de crear una matriz booleana cuyos valores (True, False)
	dependeran de que los valores RGB de la imagen original sean mayores o menores que
	los valores umbrales maximos o minimos obtenidos en la funcion sColorBalance.

	El orden de ejecucion es Blue, Green, Red
	'''
	low_mask = matrix<low_value
	matrix = apply_mask(matrix,low_mask,low_value)
#	cv2.imshow("MatrixL", matrix)

	high_mask = matrix>high_value
	matrix = apply_mask(matrix,high_mask,high_value)
#	cv2.imshow("MatrixH", matrix)

	return matrix


def sColorBalance(img, porcentaje):
	'''
	Funcion encarganda de:
	Separar los canales RGB de la imagen (split)
	Ordenar los valores de pixeles y seleccionar los "cuantiles" de la matriz ordenada
	Obtener los valores max y min de la matriz ordenanda para cada canal RGB (a partir
		del porcentaje de saturacion)
	Saturar la imagen para los valores max y min de cada canal

	Todo esto con el fin de que los colores RGB ocupen el mayor rango posible [0,255]
	aplicando una transformacion afin a cada canal
	'''

	assert img.shape[2] == 3
	#assert porcentaje > 0 and porcentaje < 100

	mitad_porcentaje = porcentaje/200.0
	canales = cv2.split(img)		#Separa los canales en RGB

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

		#Seleccion de los valores minimos y maximos de cada canal RGB de la imagen. Seria el stretching
		bajo_val  = flat[math.floor(n_cols*mitad_porcentaje)]		#Calcula el valor bajo del arreglo ordenado de la matriz (img) de entrada para cada canal
		alto_val = flat[math.ceil(n_cols*(1-mitad_porcentaje))]		#Calcula el valor alto del arreglo ordenado de la matriz (img) de entrada para cada canal 			Alternativa: alto_val = flat[math.ceil(n_cols*(1-mitad_porcentaje)-1)]

		#Los valores alto y bajo para cada canal RGB. El orden de impresion es Blue, Green, Red
		print "Lowval: ", alto_val
		print "Highval: ", bajo_val

		# saturate below the low percentile and above the high percentile
		thresholded = apply_threshold(canal,bajo_val,alto_val)
		# scale the canal
		normalized = cv2.normalize(thresholded,thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
#		cv2.imshow("Madfe", normalized)
		salida_canales.append(normalized)

	return cv2.merge(salida_canales)


if __name__ == '__main__':
	imgOriginal = cv2.imread('GOPR0535_Cap_0004.jpg')

	#-----Llamado a Funcion----------------------------------------------------
	imgRecuperada = sColorBalance(imgOriginal, 1)	#Porcentaje de umbral inferior y superior respecto al histograma de entrada. Este porcentaje puede ser distinto para c/limite del histograma

	#-----Resultados----------------------------------------------------
	cv2.namedWindow('imgOriginal',cv2.WINDOW_NORMAL)
	cv2.imshow("imgOriginal", imgOriginal)
	cv2.namedWindow('imgRecuperada',cv2.WINDOW_NORMAL)
	cv2.imshow("imgRecuperada", imgRecuperada)

	#-----Guardado de la imagen Recuperada-------------------------------------------
	cv2.imwrite('imagenRecuperadaCR_RGB.jpg',imgRecuperada)

	#-----Calculo de Histograma----------------------------------------------------
	'''
	Se calcula el histograma de la imagen original y la recuperada para analizar el
	efecto del algoritmo sobre la imagen
	'''
	color = ('b','g','r')
	for i, col in enumerate(color):
	   histcolorOriginal =  cv2.calcHist([imgOriginal],[i],None,[256],[0,256])
	   histcolorRecuperada =  cv2.calcHist([imgRecuperada],[i],None,[256],[0,256])

	   plt.subplot(211), plt.plot(histcolorOriginal, color=col)
	   plt.title('Histograma Original')
	   plt.ylabel('Numero de Pixeles')
	   plt.xlim([0,256])


	   plt.subplot(212), plt.plot(histcolorRecuperada,color=col)
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
	IMG = cv2.imread('GOPR0535_Cap_0004.jpg',0)
	IMGRec = cv2.imread('imagenRecuperadaCR_RGB.jpg',0)
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

	cv2.waitKey(0)
	cv2.destroyAllWindows()
