#!/bin/bash
# 2017-11-10

# Automatic image extraction for undarwater preprocessing images
# List, moves, and strip jpg files at user-specified frame rate.
# v 0.3

if [ -z "$1" ]; then 
	echo usage: $0 path_to_images [python_module]
        exit
fi

PATH_BASE=$1
MODULE=$2

if [ -z "$2" ]; then 
	echo "Using default value for MODULE: FeaturesDetection.py" 
    MODULE="codigoSIFT.py"
fi

echo "Filter module: " $MODULE

# Retrieves the list of all image files with .jpg, .JPG, .png extension
#$LIST="$(ls "$1"*.JPG)"  #Here put the extension of image to process

shopt -s nullglob
LIST=$(ls -d $PATH_BASE*/)
echo $LIST 

for filename in $PATH_BASE*.JPG; do
  echo "Image found -> $filename"
  COMMAND="python $MODULE -i $filename"
  echo $COMMAND
  $($COMMAND)
done

