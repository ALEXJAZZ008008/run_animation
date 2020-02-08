#!/bin/bash

#Copyright University College London 2020
#Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
#For internal research only.

INPUTPATH=${1}
OUTPUTPATH=${2}

echo -e "input path: " $INPUTPATH "\noutput path: " $OUTPUTPATH "\n"

echo -e 'remove '"$OUTPUTPATH"'\n'
rm -rf "$OUTPUTPATH"

# ----
# data
# ----
echo -e 'make '"$OUTPUTPATH"'\n'
mkdir -p "$OUTPUTPATH"

if [ -d "$INPUTPATH"'/ground_truth/cropped_xcat/' ]
then
    echo -e "make ground truth directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/ground_truth/'
    cp -r "$INPUTPATH"'/ground_truth/cropped_xcat/' "$OUTPUTPATH"'/ground_truth/'
fi

if [ -d "$INPUTPATH"'/gated_ground_truth/' ]
then
    echo -e "make only gated directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/gated_ground_truth/'
    cp -r "$INPUTPATH"'/gated_ground_truth/' "$OUTPUTPATH"'/gated_ground_truth/'
fi

if [ -d "$INPUTPATH"'/cropped_input/saveDyn/' ]
then
    echo -e "make cropped input dynamic directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/cropped_input/'
    cp -r "$INPUTPATH"'/cropped_input/saveDyn/' "$OUTPUTPATH"'/cropped_input/'
fi

if [ -d "$INPUTPATH"'/reconstructed/saveDyn/' ]
then
    echo -e "make reconstructed dynamic directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/reconstructed/'
    cp -r "$INPUTPATH"'/reconstructed/saveDyn/' "$OUTPUTPATH"'/reconstructed/'
fi
