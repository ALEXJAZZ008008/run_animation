#!/bin/bash

#Copyright University College London 2019
#Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
#For internal research only.

INPUTPATH=${1}
OUTPUTPATH=${2}

echo -e "input path: " $INPUTPATH "\noutput path: " $OUTPUTPATH "\n"

echo -e "run run migrate\n"

./run_migrate.sh $INPUTPATH $OUTPUTPATH

if [ -d "$OUTPUTPATH"'/cropped_attenuation/' ]
then
    echo -e "run run animation on cropped attenuation\n"
    python3.6 ./run_animation.py "$OUTPUTPATH"'/cropped_attenuation/' None "$OUTPUTPATH"'/cropped_attenuation_output/' "cropped_attenuation_output.gif" False None ".nii" False None None 2.1306 3.27 1 125 True 0.5
fi

if [ -d "$OUTPUTPATH"'/attenuation/' ]
then
    echo -e "run run animation on attenuation\n"
    python3.6 ./run_animation.py "$OUTPUTPATH"'/attenuation/' None "$OUTPUTPATH"'/attenuation_output/' "attenuation_output.gif" False None ".nii" False None None 2.1306 3.27 1 125 True 0.5
fi

if [ -d "$OUTPUTPATH"'/gated_ground_truth/' ]
then
    echo -e "run run animation on gated ground truth\n"
    python3.6 ./run_animation.py "$OUTPUTPATH"'/gated_ground_truth/' None "$OUTPUTPATH"'/gated_ground_truth_output/' "gated_ground_truth_output.gif" True "fixed" ".nii" True "fixed" ".nii" 2.1306 3.27 1 125 True 0.5
fi

if [ -d "$OUTPUTPATH"'/momo_cropped_input/' ]
then
    echo -e "run run animation on momo cropped input\n"
    python3.6 ./run_animation.py "$OUTPUTPATH"'/momo_cropped_input/' "$OUTPUTPATH"'/momo_dynamic_cropped_input/' "$OUTPUTPATH"'/momo_cropped_input_output/' "cropped_input_output.gif" False None ".nii" True "fixed" ".nii" 2.1306 3.27 1 133 True 0.5
fi

if [ -d "$OUTPUTPATH"'/momo_reconstructed/' ]
then
    echo -e "run run animation on reconstructed\n"
    python3.6 ./run_animation.py "$OUTPUTPATH"'/momo_reconstructed/' "$OUTPUTPATH"'/momo_dynamic_reconstructed/' "$OUTPUTPATH"'/momo_reconstructed_output/' "reconstructed_output.gif" False None ".nii" True "fixed" ".nii" 2.1306 3.27 1 133 True 0.5
fi

if [ -d "$OUTPUTPATH"'/cropped_input/' ]
then
    echo -e "run run animation on momo cropped input\n"
    python3.6 ./run_animation.py "$OUTPUTPATH"'/cropped_input/' "$OUTPUTPATH"'/ground_truth/' "$OUTPUTPATH"'/cropped_input_output/' "cropped_input_output.gif" True "GT" ".nii" False None ".nii" 2.1306 3.27 1 133 True 0.5
fi

if [ -d "$OUTPUTPATH"'/reconstructed/' ]
then
    echo -e "run run animation on reconstructed\n"
    python3.6 ./run_animation.py "$OUTPUTPATH"'/reconstructed/' "$OUTPUTPATH"'/ground_truth/' "$OUTPUTPATH"'/reconstructed_output/' "reconstructed_output.gif" True "GT" ".nii" False None ".nii" 2.1306 3.27 1 133 True 0.5
fi
