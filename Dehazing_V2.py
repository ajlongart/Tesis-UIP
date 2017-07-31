#!/usr/bin/env python
'''
Modulo 1 Toolbox
Dehazing Removal Using Dark-Channel Prior using Guided Filter
Tesis Underwater Image Pre-processing
Armando Longart 10-10844
ajzlongart@gmail.com

Descripcion: Modulo implementado para eliminar el haze (niebla, neblina...)
presente en las imagenes subacuaticas. Esta basado en el algoritmo llamado
dark-channel prior. Este trabajo esta basado en el paper "Single Image Haze 
Removal Using Dark Channel Prior" y "Guided Image Filtering" dasarrollado 
por Kaiming He y otros.

Para este algoritmo se basa en la ecuacion de la imagen con haze
I(x) = J(x)t(x)+A[1-t(x)]
I(x) es la imagen observada (capturada con la camara)
J(x) es la radiancia en la escena (imagen a recuperar)
A es la luz atmosferica
t(x) es el media de transmision

Modulo implementado en Python
'''
# Python 2/3 compatibility
import numpy as np
import cv2
from numpy import *
from matplotlib import pyplot as plt


#-----Funciones a Implementar-----------------------------------------------------
def get_dark_channel(img, tamanyo):
	'''
	Esta funcion obtiene el dark-channel prior de la imagen RGB. El dark-channel se 
	basa en el hecho de que la mayoria de los parches sin cielo (nonsky patches) al 
	menos un canal de color tiene algunos pixeles cuya intensidad es muy baja, cercana 
	a cero. Basando en lo anterior, si una imagen es libre de haze excepto en la sky 
	region la intensidad del dark-channel es baja y tiende a cero
	'''
	#FUNCION REVISADA
	filas,columnas,canales = img.shape
	pad_size = tamanyo/2 
	padded_img = np.pad(img, (pad_size, pad_size),'constant',constant_values=np.inf)
	dark_channel = np.zeros((filas,columnas))
	for i in range(1,filas):
		for j in range(1,columnas):
			dark_channel[i,j] = np.min(padded_img[i:i+tamanyo, j:j+tamanyo, :])

	'''
	El dark-channel es el resultado de 2 operadores minimos: uno sobre cada pixel
	(en el espacio de color RGB, padded_img) y el otro es un filtro minimo: dark_channel
	'''
			
	return dark_channel


def get_atmosphere(img, dark_channel):
	'''
	Esta funcion obtiene la luz atmosferica de la imagen RGB. A partir del dark-channel
	se escogen (pick) el 0.1porciento de pixeles. Estos pixeles son usualmente los
	mas opacos (most haze-opaque). Entre esos pixeles, los pixeles con una mayor 
	intensidad en la imagen de entrada se seleccionan como la luz atmosferica
	'''
	#FUNCION REVISADA
	filas,columnas,canales = img.shape
	numPixels = filas*columnas
	searchPixels = np.floor(numPixels*0.1)	#0.01	
	searchPixelsInt = int(round(searchPixels))
	dark_vector = np.reshape(dark_channel, (numPixels, 1))
	img_vector = np.reshape(img, (numPixels,3))
	print img_vector.shape 
	indices = np.sort(dark_vector,axis=0)[::-1]		#Ordenamiento descendente. O tambien usar dark_vector[::-1].sort()
	indicesA = dark_channel.ravel().argsort()[::-1]

	contador = np.zeros((1,3))	
	for i in range(1,searchPixelsInt+1):
		acum = indicesA[i]
		contador += img_vector[acum,:]	
	
	atmosphere = contador/searchPixelsInt

	return atmosphere


def get_transmission_estimate(img, atmosphere, omega, tamanyo):
	'''
	Esta funcion estima la transmision de la imagen RGB. A partir del dark-channel y de 
	la ecuacion de la imagen con haze (vease Descripcion) y si se asume que la radiancia 
	es una imagen libre de haze, se puede estimar la transmision como el usado por trans_est.
	El valor de omega se usa para mantener una nocion de profundidad en la imagen a recuperar
	'''
	#FUNCION REVISADA
	filas,columnas,canales = img.shape
	rep_atmosphere = np.tile(atmosphere,(filas,columnas,1))  
	trans_est = 1-omega*get_dark_channel(img/rep_atmosphere,tamanyo)	#Ecuacion de la estimacion de transmision	

	return trans_est


def get_boxfilter(img_gray,radius):
	'''
	Funcion auxiliar encargada de  crear el filtro de media de ventana radius
	Se aplica tanto al Eje X como al Eje Y de la imagen en escala de grises 
	'''
	#FUNCION REVISADA
	filas_gray,columnas_gray = img_gray.shape
	sum_img = np.zeros((filas_gray,columnas_gray))

	#Suma acumulativa Eje Y
	sumY = np.cumsum(img_gray, axis=0)
	#Diferencia sobre el Eje Y
	sum_img[0:radius+1, :] = sumY[radius:2*radius+1 ,:]	 #(o) sum_img[:radius+1] = sumY[radius:2*radius+1]
	sum_img[radius+1:filas_gray-radius ,:] = sumY[2*radius+1:filas_gray ,:]		#(o) sum_img[radius+1:filas_gray-radius] = sumY[2*radius+1:]-sumY[:filas_gray-2*radius-1]
	sum_img[-radius:] = np.tile(sumY[-1],(radius,1))-sumY[filas_gray-2*radius-1:filas_gray-radius-1]		#(o) sum_img[filas_gray-radius+1:filas_gray,:] = np.tile(sumY[filas_gray,:],(radius,1))-sumY[filas_gray-2*radius+1:filas_gray-radius-1,:]

	#Suma acumulativa Eje X
	sumX = np.cumsum(sum_img,axis=1)
	#Diferencia sobre el Eje X
	sum_img[:, 0:radius+1] = sumX[:, radius:2*radius+1]	#(o) sum_img[:, :radius+1] = sumX[:, radius:2*radius+1]
	sum_img[:, radius+1:columnas_gray-radius] = sumX[:,2*radius+1:columnas_gray]-sumX[:,0:columnas_gray-2*radius-1]		#(o) sum_img[:, radius+1:columnas_gray-radius] = sumX[:, 2*radius+1:]-sumX[:, :columnas_gray-2*radius-1]
	sum_img[:, -radius:] = np.tile(sumX[:,-1][:,None], (1,radius)) - sumX[:, columnas_gray-2*radius-1:columnas_gray-radius-1] 	#(o) sum_img[:, columnas_gray-radius+1:columnas_gray] = np.tile(sumX[:,columnas_gray],(1,radius))-sum_img[:,columnas_gray-2*radius:columnas_gray-radius-1] 

	return sum_img


def get_guidedfilter (img, trans_est, radius, eps):
	'''
	Esta funcion crea el filtro usado para eliminar el haze de la imagen.
	Este filtro se puede usar como un operador de suavizado de bordes con 
	un mayor desempenyo que el filtro bilateral. Este filtro tiene un algoritmo 
	de tiempo lineal rapido. La salida de este filtro es localmente una 
	transformada lineal de la imagen de guia.
	'''
	#FUNCION REVISADA
	img32bits = np.float32(img)

	img_gray = cv2.cvtColor(img32bits, cv2.COLOR_BGR2GRAY)
	filas_gray,columnas_gray, = img_gray.shape
	img_gray_vector = np.ones((filas_gray,columnas_gray))
	prom_denom = get_boxfilter(img_gray_vector,radius)		#Funcion encargada de crear el filtro de media de ventana radius

	'''
	Las variables mean, correlaccion var y covar son de significada intuitivo
	(se puede decir probabilisticos)
	'''
	mean_I = get_boxfilter(img_gray,radius)/prom_denom		#Filtro de media de la imagen en escala de gris
	mean_T = get_boxfilter(trans_est,radius)/prom_denom		#Filtro de media de la estimacion de transmision

	correlacion_I = get_boxfilter(img_gray*img_gray,radius)/prom_denom		#Correlacion de la imagen. 
	correlacion_IT = get_boxfilter(img_gray*trans_est,radius)/prom_denom	#Correlacion de la estimacion de transmision. 

	var_I = correlacion_I-mean_I*mean_I 		#Varianza de la imagen. 
	covar_IT = correlacion_IT-mean_I*mean_T 	#Covarianza de la imagen y de la estimacion

	A = covar_IT/(var_I+eps)
	B = mean_T-A*mean_I

	mean_A = get_boxfilter(A,radius)/prom_denom		#Funcion encargada de crear el filtro de media de ventana radius
	mean_B = get_boxfilter(B,radius)/prom_denom		#Funcion encargada de crear el filtro de media de ventana radius

	guidedFilter = mean_A*img_gray+mean_B			#Ecuacion de guided filter

	return guidedFilter


def get_radiance(img, transmission, atmosphere):
	'''
	Esta funcion recupera los valores de radiancia de la imagen
	original con la luz atmosferica y la estimacion de
	transmision ya conocidos. Es decir, la imagen sin el haze
	'''
	#FUNCION REVISADA
	filas,columnas,canales = img.shape
	rep_atmosphere = np.tile(atmosphere,(filas,columnas,1))
	max_tran = np.maximum(transmission,0.1)							#Se saca el maximo de la transmision para restringir
	max_transmission = np.tile(max_tran[:,:,np.newaxis],(1,1,3))

	'''
	Ecuacion de recuperacion de la imagen original J(x)={[I(x)-A]/max[t(x)-to]}+A
	'''
	radiance = ((img-rep_atmosphere)/max_transmission)+rep_atmosphere 	

	return radiance


if __name__ == '__main__':
	#-----Lectura de Imagen-----------------------------------------------------
	#Se usa el formato double para el algoritmo.
	img = double(cv2.imread('MVI_0234_Cap1.png'))/255
	#Usado para calcular el histograma y la conversion al canal YCrCb. La imagen 
	#para ambos casos debe ser o int 8bits, o int 16bits o float 32bits: cv2.cvtColor y calcHist
	imgOriginal = cv2.imread('MVI_0234_Cap1.png')
	##Para reduccion, se usa Area. Para amplicacion, (Bi)Cubica INTER_CUBIC
	img = cv2.resize(img,None, fx=0.8,fy=0.8,interpolation=cv2.INTER_AREA)
	cv2.namedWindow('img',cv2.WINDOW_NORMAL)
	cv2.imshow("img",imgOriginal)

	print imgOriginal.dtype 

	#-----Tamanyo de la Imagen----------------------------------------------------
	filas,columnas,canales = img.shape

	#-----Variables de Interes---------------------------------------------------
	omega = 0.95	#Parametro usado para conservar una minima cantidad de haze en la imagen para...  
					#...efectos de profundidad. Rango entre 0 y 1. Transmission Estimate
	tamanyo = 15	#Tamanyo del parche local (bloque) centrado un cierto pixel x. Dark-Channel Prior
	radius = 50		#Tamanyo del bloque para el Guided Filter
	eps = 0.001		#Parametro de regularizacion (llamado en el paper epsilon). Para el Guided Filter

	#-----Llamado a Funciones----------------------------------------------------
	dark_channel = get_dark_channel(img, tamanyo)
	atmosphere = get_atmosphere(img, dark_channel)
	trans_est = get_transmission_estimate(img, atmosphere, omega, tamanyo)
	filtro = get_guidedfilter(img,trans_est,radius,eps)
	transmission = np.reshape(filtro, (filas,columnas))
	radiance = get_radiance(img, transmission, atmosphere)


	#-----Resultados----------------------------------------------------
	cv2.namedWindow('darkChannel',cv2.WINDOW_NORMAL)
	cv2.imshow('darkChannel', dark_channel)
	cv2.namedWindow('atmosphere',cv2.WINDOW_NORMAL)
	cv2.imshow('atmosphere', atmosphere)
	cv2.namedWindow('transEst',cv2.WINDOW_NORMAL)
	cv2.imshow('transEst', trans_est)
	cv2.namedWindow('Filtro',cv2.WINDOW_NORMAL)
	cv2.imshow('Filtro', filtro)
	cv2.namedWindow('transmision',cv2.WINDOW_NORMAL)
	cv2.imshow('transmision', transmission)
	cv2.namedWindow('radianciaOriginal',cv2.WINDOW_NORMAL)
	cv2.imshow('radianciaOriginal', radiance)

	#-----Guardado de la imagen Recuperada-------------------------------------------
	radianceNew = cv2.resize(radiance,None, fx=1.25,fy=1.25,interpolation=cv2.INTER_CUBIC)
	radiance255 = radianceNew*255
	cv2.imwrite('imagenRecuperadaDehaze.jpg',radiance255)
	imgRecuperada = cv2.imread('imagenRecuperadaDehaze.jpg')

	#-----Separacion de canales RGB-----------------------------------------------
	Rrec, Grec, Brec = cv2.split(imgRecuperada)
	cv2.namedWindow('canalRojo',cv2.WINDOW_NORMAL)
	cv2.imshow('canalRojo', Rrec)
	cv2.namedWindow('canalVerde',cv2.WINDOW_NORMAL)
	cv2.imshow('canalVerde', Grec)
	cv2.namedWindow('canalAzul',cv2.WINDOW_NORMAL)
	cv2.imshow('canalAzul', Brec)


	#-----Comparaciones---------------------------------------------------------
	'''
	Comparaciones en las imagenes original y recuperada para observar la 
	diferencia en el canal de luminancia
	'''
	img_yrb = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2YCR_CB)		#img_yrb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	YOri, CrOri, CbOri = cv2.split(img_yrb)

	radiance_yrb = cv2.cvtColor(imgRecuperada, cv2.COLOR_BGR2YCR_CB)		#radiance_yrb = cv2.cvtColor(radiance, cv2.COLOR_BGR2YCrCb)
	Yrec, Crrec, Cbrec = cv2.split(radiance_yrb)

	cv2.namedWindow('img YCrCb',cv2.WINDOW_NORMAL)
	cv2.imshow('img YCrCb', YOri)
	cv2.namedWindow('radiancia img YCrCb',cv2.WINDOW_NORMAL)
	cv2.imshow('radiancia img YCrCb', Yrec)

	YOri_32bits = np.float32(YOri)
	Yrec_32bits = np.float32(Yrec)
	resta = cv2.subtract(YOri_32bits,Yrec_32bits)
	cv2.namedWindow('Resta',cv2.WINDOW_NORMAL)
	cv2.imshow('Resta', resta)


	#-----Calculo de Histograma----------------------------------------------------
	'''
	Se calcula el histograma de la imagen con haze, la imagen recuperada (c/u en el
	espacio RGB) y el canal de luminancia de c/u con el objeto de analizar los resultados
	del algoritmo
	'''
	color = ('b','g','r')
	for i, col in enumerate(color):
	   histcolorOriginal =  cv2.calcHist([imgOriginal],[i],None,[256],[0,256])
	   histcolorOriginal_Y =  cv2.calcHist([YOri],[0],None,[256],[0,256])
	   histcolorRecuperada =  cv2.calcHist([imgRecuperada],[i],None,[256],[0,256])
	   histcolorRecuperada_Y =  cv2.calcHist([Yrec],[0],None,[256],[0,256])
	   
	   cv2.normalize(histcolorOriginal,histcolorOriginal,8,cv2.NORM_MINMAX)
	   cv2.normalize(histcolorOriginal_Y,histcolorOriginal_Y,8,cv2.NORM_MINMAX)
	   cv2.normalize(histcolorRecuperada,histcolorRecuperada,8,cv2.NORM_MINMAX)
	   cv2.normalize(histcolorRecuperada_Y,histcolorRecuperada_Y,8,cv2.NORM_MINMAX)

	   plt.subplot(221), plt.plot(histcolorOriginal, color=col)
	   plt.title('Histograma Original')
	   plt.ylabel('Numero de Pixeles')
	   plt.xlim([0,256])

	   plt.subplot(222), plt.plot(histcolorOriginal_Y)
	   plt.title('Histograma Luminancia Original')
	   plt.xlim([0,256])

	   plt.subplot(223), plt.plot(histcolorRecuperada,color=col)
	   plt.title('Histograma Recuperada')
	   plt.ylabel('Numero de Pixeles')
	   plt.xlabel('Bins')
	   plt.xlim([0,256])

	   plt.subplot(224), plt.plot(histcolorRecuperada_Y)
	   plt.title('Histograma Luminancia Recuperada')
	   plt.xlabel('Bins')
	   plt.xlim([0,256])

	plt.show()

	cv2.waitKey()
	cv2.destroyAllWindows()

