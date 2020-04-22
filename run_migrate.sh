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

if [ -d "$INPUTPATH"'/cropped_attenuation/' ]
then
    echo -e "make cropped attenuation directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/cropped_attenuation/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/cropped_attenuation/.' "$OUTPUTPATH"'/cropped_attenuation/'
fi

if [ -d "$INPUTPATH"'/attenuation/' ]
then
    echo -e "make attenuation directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/attenuation/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/attenuation/.' "$OUTPUTPATH"'/attenuation/'
fi

if [ -d "$INPUTPATH"'/gated_ground_truth/dynamic/' ]
then
    echo -e "make only gated directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/gated_ground_truth/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/gated_ground_truth/dynamic/.' "$OUTPUTPATH"'/gated_ground_truth/'
fi

if [ -d "$INPUTPATH"'/cropped_input/dynamic_volumes/dynamic/' ]
then
    echo -e "make momo dynamic volume cropped input dynamic directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/momo_dynamic_cropped_input/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/cropped_input/dynamic_volumes/dynamic/.' "$OUTPUTPATH"'/momo_dynamic_cropped_input/'
fi

if [ -d "$INPUTPATH"'/reconstructed/dynamic_volumes/dynamic/' ]
then
    echo -e "make momo dynamic volume reconstructed dynamic directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/momo_dynamic_reconstructed/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/reconstructed/dynamic_volumes/dynamic/.' "$OUTPUTPATH"'/momo_dynamic_reconstructed/'
fi

if [ -d "$INPUTPATH"'/cropped_input/saveDyn/' ]
then
    echo -e "make momo cropped input dynamic directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/momo_cropped_input/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/cropped_input/saveDyn/.' "$OUTPUTPATH"'/momo_cropped_input/'
fi

if [ -d "$INPUTPATH"'/reconstructed/saveDyn/' ]
then
    echo -e "make momo reconstructed dynamic directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/momo_reconstructed/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/reconstructed/saveDyn/.' "$OUTPUTPATH"'/momo_reconstructed/'
fi

if [ -d "$INPUTPATH"'/ground_truth/cropped_xcat/' ]
then
    echo -e "make ground truth directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/ground_truth/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/ground_truth/cropped_xcat/.' "$OUTPUTPATH"'/ground_truth/'
fi

if [ -d "$INPUTPATH"'/evaluation/comparison_input/' ]
then
    echo -e "make cropped input dynamic directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/cropped_input/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/evaluation/comparison_input/.' "$OUTPUTPATH"'/cropped_input/'
fi

if [ -d "$INPUTPATH"'/evaluation/comparison_reconstructed/' ]
then
    echo -e "make reconstructed dynamic directory and move results\n"
    mkdir -p "$OUTPUTPATH"'/reconstructed/'
    rsync -vaOzhP --no-p --no-g --no-o --append-verify --inplace "$INPUTPATH"'/evaluation/comparison_reconstructed/.' "$OUTPUTPATH"'/reconstructed/'
fi
