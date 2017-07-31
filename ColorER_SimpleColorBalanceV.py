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
mejorada. Se uso el modelo de color HSV para el algoritmo. La diferencia
de este modulo con el SimpleColorBalance en HSV es que este solo lo hace 
para el canal V (value) debido a que este indica cuan brillante es o no 
una imagen. Los otros dos canales H (Hue) y S (Saturation) generan cambios
sustanciales en la imagen de salida adulterando el color y agregando blancos
respectivamente alterando los resultados.

Modulo implementado en Python

Variacion del Simple Color Balance (en RGB) de DavidYKay: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc 
'''
# Python 2/3 compatibility
import cv2
import math
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import colors

#-----Funciones a Implementar-----------------------------------------------------
def apply_mask(matrix, mask, fill_value):
	'''
	Funcion encargada de "crear" la matriz de valores enmascarados. Estos valores son
	determinados a partir de los valores altos y bajos del la funcion apply_threshold
	(low_mask, high-mask como mask y de low_value, high_value como fill_value).

	El orden de ejecucion es Hue, Saturation, Value
	'''
	masked = np.ma.array(matrix,mask=mask,fill_value=fill_value)
	cv2.imshow("Masked", masked)
	cv2.imshow("MaskFill", masked.filled())
	cv2.imshow("MaskedFill", masked.filled([0]))
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
	cv2.imshow("MatrixL", matrix)

	high_mask = matrix>high_value
	matrix = apply_mask(matrix,high_mask,high_value)
	cv2.imshow("MatrixH", matrix)

	return matrix


def sColorBalance(img_hsv, porcentaje):
	'''
	Funcion encarganda de:
	Separar los canales HSV de la imagen (split)
	Ordenar los valores de pixeles y seleccionar los "cuantiles" de la matriz ordenada
	Obtener los valores max y min de la matriz ordenanda para el canal V (a partir
		del porcentaje de saturacion)
	Saturar la imagen para los valores max y min de cada canal

	Todo esto con el fin de que los colores de la imagen recuperada ocupen el mayor rango posible 
	[0,255] aplicando una transformacion solo al canal V
	'''
	assert img_hsv.shape[2] == 3
	assert porcentaje > 0 and porcentaje < 100

	mitad_porcentaje = porcentaje/200.0
	hueOri,satOri,valOri = cv2.split(img_hsv)		#Separa los canales en HSV
	filas,columnas,canales = img_hsv.shape
	
	cv2.imshow("h", hueOri)
	cv2.imshow("s", satOri)
	cv2.imshow("v", valOri)

	salida_canales = []
	canal = valOri

	assert len(canal.shape) == 2
	print canal
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
	print normalized
	cv2.imshow("Madfe", normalized)
	salida_canales.append(normalized)
	img_merge = cv2.merge((hueOri,satOri,normalized))

	return img_merge


if __name__ == '__main__':
	imgOriginal = cv2.imread('MVI_0234_Cap1.png')
	img_hsv = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV) 		#Conversion de HSV de la imagen original RGB

	#-----Llamado a Funcion----------------------------------------------------
	imgRecuperada = sColorBalance(img_hsv, 1)	#Porcentaje de umbral inferior y superior respecto al histograma de entrada. Este porcentaje puede ser distinto para c/limite del histograma
	imgRecuperadaRGB = cv2.cvtColor(imgRecuperada,cv2.COLOR_HSV2BGR)	#Conversion del canal V de la imagen recuperada a RGB

	#-----Resultados----------------------------------------------------
	cv2.imshow("imgOriginal", imgOriginal)
	cv2.imshow("imgRecuperadaHSV", imgRecuperada)
	cv2.imshow("imgRecuperada", imgRecuperadaRGB)

	#-----Guardado de la imagen Recuperada-------------------------------------------
	cv2.imwrite('imagenRecuperadaCR_V_RGB.jpg',imgRecuperadaRGB)

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

	

	cv2.waitKey(0)
	cv2.destroyAllWindows()
