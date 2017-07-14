# Tesis-UIP
Módulos de Pre-procesamiento de Imágenes para aplicaciones subacuáticas

El primero de estos modulos es el Dehazing, el cual se puede traducir como eliminación de 
niebla en las imágenes subacuáticas. Está basado en el algoritmo llamado dark-channel prior 
y en el guided filter, el cual es usado para eliminar el haze de la imagen.

El segundo de estos módulos es el Color Restoration. Actualmente se están trabajando en 2 
algoritmos: GrayWorld assumption y Simplest Color Balance. Ambos tienen como finalidad
recuperar el color rojo el cual no está presente en las imágenes subacuáticas debido a las 
propiedades (absorption, scattering...) de la luz en el agua.

Modulos en desarrollo...

Este repositorio (y la tesis en general) se hará usando OpenCV con Python
