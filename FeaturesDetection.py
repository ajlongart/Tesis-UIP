import numpy as np
import cv2
from numpy import *
import argparse
import os

'''
Analisis de Resultados
Features Detection
Tesis Underwater Image Pre-processing
Armando Longart 10-10844
ajzlongart@gmail.com

Descripcion: Modulo desarrollado para la detección de features de las imagenes 
usadas en la tesis. Se usaran 4 tipos de features (SIFT, SURF, ORB y KAZE) los 
cuales seran aplicadas a las imagenes originales y recuperadas. De este modulo
se generan archivos con la cantidad de features detectados (para cada tipo de
detector) y la ubicacion de los mismos en sus coordenadas X y Y.
Tercera herramienta para el analisis de resultados de las imagenes recuperadas.
Las dos anteriores estan en los algoritmos usados para la mejora de las imagenes

Modulo implementado en Python
'''

#Constuccion del parse y del argumento
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imageOri", required = True, help = "Imagen de Entrada")
args = vars(ap.parse_args())

#-------------------Creacion de archivos con los resultados--------------------------
f = open('features_txt.txt','a') 
fimgSIFT = open("%s_SIFT.txt" %(args["imageOri"]), "w")
fimgSURF = open("%s_SURF.txt" %(args["imageOri"]), "w")
fimgORB = open("%s_ORB.txt" %(args["imageOri"]), "w")
fimgKAZE = open("%s_KAZE.txt" %(args["imageOri"]), "w")

#-------------------Lectura de Imagenes y conversion a escala de grises--------------------------
img1 = cv2.imread(args["imageOri"])

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#-------------------Algoritmo SIFT--------------------------
#Initiate SIFT detector (with xfeatures2d modules)
sift = cv2.xfeatures2d.SIFT_create()

#Find and compute the descriptors with SIFT
kps, descs = sift.detectAndCompute(gray1, None)
print("# kps1: {}, descriptors1: {}".format(len(kps), descs.shape))

#Draw only keypoints location,not size and orientation
img1 = cv2.drawKeypoints(gray1,kps,img1)	#flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

#Show the result in new window
cv2.namedWindow('imgSIFT',cv2.WINDOW_NORMAL)
cv2.imshow("imgSIFT", img1)

#Write in the created file the coordinates X y Y of the image
for i_sift,keypoint_sift in enumerate(kps):
    #print "Keypoint %d: %s" % (i, keypoint.pt)
    fimgSIFT.write('%s \t %s \n' %(i_sift, keypoint_sift.pt))
fimgSIFT.close()

#-------------------Algoritmo SURF--------------------------
#Initiate SURF detector (with xfeatures2d modules)
surf = cv2.xfeatures2d.SURF_create()

#Find and compute the descriptors with SURF
kpf, desf = surf.detectAndCompute(gray1,None)
print("# kpf: {}, descriptorsf: {}".format(len(kpf), desf.shape))

#Draw only keypoints location,not size and orientation
imgf = cv2.drawKeypoints(gray1,kpf,img1)

#Show the result in new window
cv2.namedWindow('imgSURF',cv2.WINDOW_NORMAL)
cv2.imshow("imgSURF", imgf)

#Write in the created file the coordinates X y Y of the image
for i_surf,keypoint_surf in enumerate(kpf):
    fimgSURF.write('%s \t %s \n' %(i_surf, keypoint_surf.pt))
fimgSURF.close()

#-------------------Algoritmo ORB--------------------------
#Initiate ORB detector
orb = cv2.ORB_create()

#Find the keypoints with ORB
kpo = orb.detect(gray1,None)

#Compute the descriptors with ORB
kpo, deso = orb.compute(gray1, kpo)
print("# kpo: {}, descriptorso: {}".format(len(kpo), deso.shape))

#Draw only keypoints location,not size and orientation
imgo = cv2.drawKeypoints(gray1,kpo,img1)

#Show the result in new window
cv2.namedWindow('imgORB',cv2.WINDOW_NORMAL)
cv2.imshow("imgORB", imgo)

#Write in the created file the coordinates X y Y of the image
for i_orb,keypoint_orb in enumerate(kpo):
    fimgORB.write('%s \t %s \n' %(i_orb, keypoint_orb.pt))
fimgORB.close()

#-------------------Algoritmo KAZE--------------------------
#Initiate KAZE detector
kaze = cv2.KAZE_create()

#Find the keypoints with KAZE
kpk = kaze.detect(gray1, None)

#Compute the descriptors with KAZE
kpk, desk = kaze.compute(gray1, kpk)
print("# kpk: {}, descriptorsk: {}".format(len(kpk), desk.shape))

#Draw only keypoints location,not size and orientation
imgk = cv2.drawKeypoints(gray1,kpk,img1)

#Show the result in new window
cv2.namedWindow('imgKAZE',cv2.WINDOW_NORMAL)
cv2.imshow("imgKAZE", imgk)

#Write in the created file the coordinates X y Y of the image 
for i_kaze,keypoint_kaze in enumerate(kpk):
    fimgKAZE.write('%s \t %s \n' %(i_kaze, keypoint_kaze.pt))
fimgKAZE.close()

#-----Escritura del archivo con los resultados----------------------------------------------
#Con write()
f.write('%s \t %d \t %d \t %d \t %d \n' %(args["imageOri"], len(kps), len(kpf), len(kpo), len(kpk)))
f.close()

#fimg.write('%s \n' %(kps1))
#fimg.close()

#cv2.imwrite('sift_keypoints19.jpg',img1)
#cv2.imwrite('sift_keypoints20.jpg',img2)

cv2.waitKey()
cv2.destroyAllWindows()
