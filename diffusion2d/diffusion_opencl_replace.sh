#!/bin/bash

DATA=`sh diffusion_opencl_neighbor_generator.sh $1`
awk -v r="$DATA" '{gsub(/#BORDER_CONDITIONS/,r)}1' diffusion_opencl_base.cl > diffusion_opencl.cl
