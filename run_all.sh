#!/bin/bash

#Copyright University College London 2019
#Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
#For internal research only.

INPUTPATH=${1}
OUTPUTPATH=${2}

echo -e "input path: " $INPUTPATH "\noutput path: " $OUTPUTPATH "\n"

echo -e "run run migrate\n"

./run_migrate.sh $INPUTPATH $OUTPUTPATH
