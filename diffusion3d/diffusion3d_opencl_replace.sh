#!/bin/bash

DATA=`sh diffusion3d_opencl_neighbor_generator.sh $1`
awk -v r="$DATA" '{gsub(/#BORDER_CONDITIONS/,r)}1' diffusion3d_opencl_base.cl > diffusion3d_opencl.cl
