#!/bin/bash

runs=5
size=512
iter=1000
block_x=(32)
block_y=(8)
block_z=(10)

for i in "${block_x[@]}"
do
	for j in "${block_y[@]}"
	do
		for k in "${block_z[@]}"
		do
			echo "$i" | xargs printf "%-5s"
			echo "$j" | xargs printf "%-5s"
			echo "$k" | xargs printf "%-5s"

			timesum=0
			flopssum=0
			bytessum=0
			energysum=0
			powersum=0

			make clean >/dev/null 2>&1; make diffusion3d_cuda_shared6.exe BLOCK_X="$i" BLOCK_Y="$j" GRID_Z="$k" >/dev/null 2>&1
			for (( l=1; l<=$runs; l++ ))
			do
				out=`./diffusion3d_cuda_shared6.exe --size $size --count $iter 2>&1`

				accuracy=`echo "$out" | grep Accuracy | cut -d " " -f 7`
				time=`echo "$out" | grep "Kernel-only" -A 5 | grep time | cut -d " " -f 4`
				flops=`echo "$out" | grep "Kernel-only" -A 5 | grep FLOPS | cut -d " " -f 10`
				bytes=`echo "$out" | grep "Kernel-only" -A 5 | grep Throughput | cut -d " " -f 5`
				energy=`echo "$out" | grep "Kernel-only" -A 5 | grep Energy | cut -d " " -f 4`
				power=`echo "$out" | grep "Kernel-only" -A 5 | grep power | cut -d " " -f 3`

				timesum=`echo $timesum+$time | bc -l`
				flopssum=`echo $flopssum+$flops | bc -l`
				bytessum=`echo $bytessum+$bytes | bc -l`
				energysum=`echo $energysum+$energy | bc -l`
				powersum=`echo $powersum+$power | bc -l`
			done

			timeaverage=`echo $timesum/$runs | bc -l`
			flopsaverage=`echo $flopssum/$runs | bc -l`
			bytesaverage=`echo $bytessum/$runs | bc -l`
			energyaverage=`echo $energysum/$runs | bc -l`
			poweraverage=`echo $powersum/$runs | bc -l`

			echo $accuracy | xargs printf "%-15s"
			echo $timeaverage | xargs printf "%-10.3f"
			echo $flopsaverage | xargs printf "%-10.3f"
			echo $bytesaverage | xargs printf "%-10.3f"
			echo $energyaverage | xargs printf "%-10.3f"
			echo $poweraverage | xargs printf "%-10.3f"
			echo
		done
	done
done
