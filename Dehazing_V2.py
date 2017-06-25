#!/usr/bin/env python
'''
Modulo 1 Toolbox
Dehazing Removal Using Dark-Channel Prior using Guided Filter
Tesis Underwater Image Pre-processing
Armando Longart 10-10844
ajzlongart@gmail.com

Descripcion: Modulo implementado para eliminar el haze (niebla, neblina...)
presente en las imagenes subacuaticas. Esta basado en el algoritmo llamado
dark-channel prior. Esta es una primera aproximacion. Faltan detalles
'''
# Python 2/3 compatibility
import numpy as np
import cv2
import scipy
from numpy import *


#-----Funciones a Implementar-----------------------------------------------------
def get_dark_channel(img, tamanyo):
	'''
	Esta funcion obtiene el dark-channel prior
	de la imagen RGB
	'''
	#FUNCION REVISADA
	filas,columnas,canales = img.shape
	pad_size = tamanyo/2 #np.floor(tamanyo/2)
	padded_img = np.pad(img, (pad_size, pad_size),'constant',constant_values=np.inf)
	dark_channel = np.zeros((filas,columnas))
	for i in range(1,filas):
		for j in range(1,columnas):
			dark_channel[i,j] = np.min(padded_img[i:i+tamanyo, j:j+tamanyo, :])
			
	return dark_channel


def get_atmosphere(img, dark_channel):
	'''
	Esta funcion obtiene la luz atmosferica 
	de la imagen RGB
	'''
	#FUNCION REVISADA
	filas,columnas,canales = img.shape
	numPixels = filas*columnas
	searchPixels = np.floor(numPixels*0.01)
	searchPixelsInt = int(round(searchPixels))
	dark_vector = np.reshape(dark_channel, (numPixels, 1))
	img_vector = np.reshape(img, (numPixels,3)) #img_vec = cv2.resize(img,None,fx=numPixels,fy=3,interpolation=cv2.INTER_CUBIC)
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
	Esta funcion obtiene la estimacion de transmision
	de la imagen RGB	
	'''
	#FUNCION REVISADA
	filas,columnas,canales = img.shape
	rep_atmosphere = np.tile(atmosphere,(filas,columnas,1)) #np.resize(atmosphere,(filas,columnas,canales)) 
	trans_est = 1-omega*get_dark_channel(img/rep_atmosphere,tamanyo)

	return trans_est


def get_boxfilter(img_gray,radius):
	'''
	Funcion auxiliar encargada de... 
	'''
	#FUNCION REVISADA
	filas_gray,columnas_gray = img_gray.shape
	sum_img = np.zeros((filas_gray,columnas_gray))

	#Eje Y
	sumY = np.cumsum(img_gray, axis=0)
	sum_img[0:radius+1, :] = sumY[radius:2*radius+1 ,:]	 #(o) sum_img[:radius+1] = sumY[radius:2*radius+1]
	sum_img[radius+1:filas_gray-radius ,:] = sumY[2*radius+1:filas_gray ,:]		#(o) sum_img[radius+1:filas_gray-radius] = sumY[2*radius+1:]-sumY[:filas_gray-2*radius-1]
	sum_img[-radius:] = np.tile(sumY[-1],(radius,1))-sumY[filas_gray-2*radius-1:filas_gray-radius-1]		#(o) sum_img[filas_gray-radius+1:filas_gray,:] = np.tile(sumY[filas_gray,:],(radius,1))-sumY[filas_gray-2*radius+1:filas_gray-radius-1,:]

	#Eje X
	sumX = np.cumsum(sum_img,axis=1)
	sum_img[:, 0:radius+1] = sumX[:, radius:2*radius+1]	#(o) sum_img[:, :radius+1] = sumX[:, radius:2*radius+1]
	sum_img[:, radius+1:columnas_gray-radius] = sumX[:,2*radius+1:columnas_gray]-sumX[:,0:columnas_gray-2*radius-1]		#(o) sum_img[:, radius+1:columnas_gray-radius] = sumX[:, 2*radius+1:]-sumX[:, :columnas_gray-2*radius-1]
	sum_img[:, -radius:] = np.tile(sumX[:,-1][:,None], (1,radius)) - sumX[:, columnas_gray-2*radius-1:columnas_gray-radius-1] 	#(o) sum_img[:, columnas_gray-radius+1:columnas_gray] = np.tile(sumX[:,columnas_gray],(1,radius))-sum_img[:,columnas_gray-2*radius:columnas_gray-radius-1] 

	return sum_img


def get_guidedfilter (img, trans_est, radius, eps):
	'''
	Esta funcion crea el filtro con el que se elimina el haze.
	Este filtro...
	'''
	#FUNCION REVISADA
	img8bits = np.array(img,dtype=np.uint8)
	img16bits = np.uint16(img)
	img32bits = np.float32(img)

	print img.dtype
	print img8bits.dtype
	print img16bits.dtype
	print img32bits.dtype

	img_gray = cv2.cvtColor(img32bits, cv2.COLOR_BGR2GRAY)
	filas_gray,columnas_gray, = img_gray.shape
	img_gray_vector = np.ones((filas_gray,columnas_gray))
	prom_denom = get_boxfilter(img_gray_vector,radius)

	mean_I = get_boxfilter(img_gray,radius)/prom_denom
	mean_T = get_boxfilter(trans_est,radius)/prom_denom

	correlacion_I = get_boxfilter(img_gray*img_gray,radius)/prom_denom
	correlacion_IT = get_boxfilter(img_gray*trans_est,radius)/prom_denom

	var_I = correlacion_I-mean_I*mean_I
	covar_IT = correlacion_IT-mean_I*mean_T

	A = covar_IT/(var_I+eps)
	B = mean_T-A*mean_I

	mean_A = get_boxfilter(A,radius)/prom_denom
	mean_B = get_boxfilter(B,radius)/prom_denom

	guidedFilter = mean_A*img_gray+mean_B

	return guidedFilter


def get_radiance(img, transmission, atmosphere):
	'''
	Esta funcion recupera los valores de radiancia de la imagen
	original con la luz atmosferica y la estimacion de
	transmision. Es decir, la imagen sin el haze
	'''
	#FUNCION REVISADA
	filas,columnas,canales = img.shape
	rep_atmosphere = np.tile(atmosphere,(filas,columnas,1))	#rep_atmosphere = np.tile(np.reshape(atmosphere,(1,1,3)),(filas,columnas))
	max_tran = np.max(transmission,0.1)
	max_transmission = np.tile(max_tran[:,None],(filas,1,3))
	
	radiance = ((img-rep_atmosphere)/max_transmission)+rep_atmosphere 	#radiance = ((img-atmosphere)/max_transmission)+atmosphere 

	return radiance


if __name__ == '__main__':
	#import sys

    #if len(sys.argv)>1:
    #    fname = sys.argv[1]
    #else :
    #    fname = '../data/lena.jpg'
    #    print("usage : python hist.py <image_file>")

    #im = cv2.imread(fname)

    #if im is None:
    #    print('Failed to load image file:', fname)
    #    sys.exit(1)

	#-----Lectura de Imagen-----------------------------------------------------
	img = double(cv2.imread('haze2.jpg'))/255
#	img = cv2.imread('forest.jpg')
	img = cv2.resize(img,None, fx=0.4,fy=0.4,interpolation=cv2.INTER_AREA)
	cv2.namedWindow('img',cv2.WINDOW_NORMAL)
	cv2.imshow("img",img) 

	#-----Tamanyo de la Imagen----------------------------------------------------
	filas,columnas,canales = img.shape

	#-----Variables de Interes. Preguntar porque---------------------------------------------------
	omega = 0.95
	tamanyo = 20
	radius = 20
	eps = 0.00001

	#-----Llamado a Funciones----------------------------------------------------
	dark_channel = get_dark_channel(img, tamanyo)
	atmosphere = get_atmosphere(img, dark_channel)
	trans_est = get_transmission_estimate(img, atmosphere, omega, tamanyo)
	filtro = get_guidedfilter(img,trans_est,radius,eps)
	transmission = np.reshape(filtro, (filas,columnas))			#DUDA si es lo mismo que cv2.resize(x,(filas, columnas))
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
	cv2.waitKey()
	cv2.destroyAllWindows()
