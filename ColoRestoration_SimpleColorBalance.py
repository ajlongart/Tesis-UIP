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
mejorada...


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
	cv2.imshow("Masked", masked)
	return masked.filled()


def apply_threshold(matrix, low_value, high_value):
	'''
	Esta funcion se encarga de crear una matriz booleana cuyos valores (True, False)
	dependeran de que los valores RGB de la imagen original sean mayores o menores que
	los valores umbrales maximos o minimos obtenidos en la funcion sColorBalance.

	El orden de ejecucion es Blue, Green, Red
	'''
	low_mask = matrix<low_value
#	print low_mask
	matrix = apply_mask(matrix,low_mask,low_value)
	cv2.imshow("MatrixL", matrix)

	high_mask = matrix>high_value
#	print high_mask
	matrix = apply_mask(matrix,high_mask,high_value)
	cv2.imshow("MatrixH", matrix)

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
#	print img.shape
	assert img.shape[2] == 3
	#assert porcentaje > 0 and porcentaje < 100

	mitad_porcentaje = porcentaje/200.0
	canales = cv2.split(img)		#Separa los canales en RGB

#	print canales

	salida_canales = []
	for canal in canales:
		assert len(canal.shape) == 2
#		print canal.shape
		# find the low and high precentile values (based on the input percentile)
		filas,columnas = canal.shape
		vec_tam = columnas*filas
		flat = canal.reshape(vec_tam)

#		print flat
#		print flat.shape

		assert len(flat.shape) == 1

		flat = np.sort(flat)

#		print flat

		n_cols = flat.shape[0]

#		print n_cols

		#Seleccion de los valores minimos y maximos de cada canal RGB de la imagen. Seria el stretching
		bajo_val  = flat[math.floor(n_cols*mitad_porcentaje)]		#Calcula el valor bajo del arreglo ordenado de la matriz (img) de entrada para cada canal
		alto_val = flat[math.ceil(n_cols*(1-mitad_porcentaje))]		#Calcula el valor alto del arreglo ordenado de la matriz (img) de entrada para cada canal 			Alternativa: alto_val = flat[math.ceil(n_cols*(1-mitad_porcentaje)-1)]

		#Los valores alto y bajo para cada canal RGB. El orden de impresion es Blue, Green, Red
		print "Lowval: ", alto_val
		print "Highval: ", bajo_val

#		print math.floor(n_cols*mitad_porcentaje)

#		print math.ceil(n_cols*(1.0-mitad_porcentaje))

		# saturate below the low percentile and above the high percentile
		thresholded = apply_threshold(canal,bajo_val,alto_val)
		# scale the canal
		normalized = cv2.normalize(thresholded,thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
		cv2.imshow("Madfe", normalized)
		salida_canales.append(normalized)

	return cv2.merge(salida_canales)


if __name__ == '__main__':
	imgOriginal = cv2.imread('MVI_0234_Cap1.png')

	#-----Llamado a Funcion----------------------------------------------------
	imgRecuperada = sColorBalance(imgOriginal, 1)	#Porcentaje de umbral inferior y superior respecto al histograma de entrada. Este porcentaje puede ser distinto para c/limite del histograma

	#-----Resultados----------------------------------------------------
	cv2.imshow("imgOriginal", imgOriginal)
	cv2.imshow("imgRecuperada", imgRecuperada)

	#-----Guardado de la imagen Recuperada-------------------------------------------
	cv2.imwrite('imagenRecuperadaCR.jpg',imgRecuperada)

	#-----Calculo de Histograma----------------------------------------------------
	'''
	Se calcula el histograma de la imagen original y la recuperada para analizar el
	efecto del algoritmo sobre la imagen
	'''
	color = ('b','g','r')
	for i, col in enumerate(color):
	   histcolorOriginal =  cv2.calcHist([imgOriginal],[i],None,[256],[0,256])
	   histcolorRecuperada =  cv2.calcHist([imgRecuperada],[i],None,[256],[0,256])

	   
	   cv2.normalize(histcolorOriginal,histcolorOriginal,8,cv2.NORM_MINMAX)
	   cv2.normalize(histcolorRecuperada,histcolorRecuperada,8,cv2.NORM_MINMAX)

	   plt.subplot(211), plt.plot(histcolorOriginal, color=col)
	   plt.title('Histograma Original')
	   plt.ylabel('Valores Normalizados')
	   plt.xlim([0,256])


	   plt.subplot(212), plt.plot(histcolorRecuperada,color=col)
	   plt.title('Histograma Recuperada')
	   plt.ylabel('Valores Normalizados')
	   plt.xlabel('Bins')
	   plt.xlim([0,256])


	plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()
