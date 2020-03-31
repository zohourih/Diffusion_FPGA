#!/bin/bash

export CL_CONTEXT_COMPILER_MODE_ALTERA=3

runs=5
iter=1000
input_size=640
no_inter=""
folder=arria10
board=p385a_sch_ax115

echo kernel | xargs printf "%-65s"
echo freq | xargs printf "%-10s"
echo last_x | xargs printf "%-10s"
echo last_y | xargs printf "%-10s"
echo time | xargs printf "%-10s"
echo flops | xargs printf "%-10s"
echo bytes | xargs printf "%-10s"
if [[ "$board" == "p385a_sch_ax115" ]]
then
	echo energy | xargs printf "%-10s"
	echo power | xargs printf "%-10s"
fi
echo

for i in `ls $folder | grep aocx | sort -V`
do
	name="${i%.*}"
	echo "$name" | xargs printf "%-65s"
	RAD=`echo $name | cut -d "_" -f 2 | cut -c 4-`
	TIME=`echo $name | cut -d "_" -f 3 | cut -c 5-`
	ASIZE=`echo $name | cut -d "_" -f 4 | cut -c 6-`
	BSIZE=`echo $name | cut -d "_" -f 5 | cut -c 6-`
	freq=`cat $folder/$name/acl_quartus_report.txt | grep Actual | cut -d " " -f 4`
    no_inter=""
	if [[ -n `echo $name | grep nointer` ]]
	then
		no_inter="NO_INTERLEAVE=1"
	fi
	if [[ -z `echo $BSIZE | grep x` ]]
	then
		BLOCK_X=$BSIZE
		BLOCK_Y=$BSIZE
	else
		BLOCK_X=`echo $BSIZE | cut -d "x" -f 1`
		BLOCK_Y=`echo $BSIZE | cut -d "x" -f 2`
	fi

	compute_bsize_x=$(($BLOCK_X - (2 * $RAD * $TIME)))
	compute_bsize_y=$(($BLOCK_Y - (2 * $RAD * $TIME)))
	last_col_x=$(($input_size + $compute_bsize_x - $input_size % $compute_bsize_x))
	last_col_y=$(($input_size + $compute_bsize_y - $input_size % $compute_bsize_y))

	timesum=0
	flopssum=0
	bytessum=0
	energysum=0
	powersum=0

	make clean >/dev/null 2>&1; make altera-host BOARD=$board RAD=$RAD TIME=$TIME ASIZE=$ASIZE BLOCK_X=$BLOCK_X BLOCK_Y=$BLOCK_Y $no_inter >/dev/null 2>&1
	rm diffusion3d_opencl.aocx >/dev/null 2>&1
	ln -s "$folder/$i" diffusion3d_opencl.aocx
	aocl program acl0 diffusion3d_opencl.aocx >/dev/null 2>&1

	for (( k=1; k<=$runs; k++ ))
	do
		out=`CL_DEVICE_TYPE=CL_DEVICE_TYPE_ACCELERATOR ./diffusion3d_altera.exe --nx $last_col_x --ny $last_col_y --nz $last_col_x --count $iter 2>&1`
		#echo "$out" >> ast.txt
		time=`echo "$out" | grep "Kernel-only" -A 5 | grep time | cut -d " " -f 4`
		flops=`echo "$out" | grep "Kernel-only" -A 5 | grep FLOPS | cut -d " " -f 10`
		bytes=`echo "$out" | grep "Kernel-only" -A 5 | grep Throughput | cut -d " " -f 5`
		if [[ "$board" == "p385a_sch_ax115" ]]
		then
			energy=`echo "$out" | grep "Kernel-only" -A 5 | grep Energy | cut -d " " -f 4`
			power=`echo "$out" | grep "Kernel-only" -A 5 | grep power | cut -d " " -f 3`
		fi

		timesum=`echo $timesum+$time | bc -l`
		flopssum=`echo $flopssum+$flops | bc -l`
		bytessum=`echo $bytessum+$bytes | bc -l`
		if [[ "$board" == "p385a_sch_ax115" ]]
		then
			energysum=`echo $energysum+$energy | bc -l`
			powersum=`echo $powersum+$power | bc -l`
		fi
	done

	timeaverage=`echo $timesum/$runs | bc -l`
	flopsaverage=`echo $flopssum/$runs | bc -l`
	bytesaverage=`echo $bytessum/$runs | bc -l`
	if [[ "$board" == "p385a_sch_ax115" ]]
	then
		energyaverage=`echo $energysum/$runs | bc -l`
		poweraverage=`echo $powersum/$runs | bc -l`
	fi

	echo $freq | xargs printf "%-10.3f"
	echo $last_col_x | xargs printf "%-10d"
	echo $last_col_y | xargs printf "%-10d"
	echo $timeaverage | xargs printf "%-10.3f"
	echo $flopsaverage | xargs printf "%-10.3f"
	echo $bytesaverage | xargs printf "%-10.3f"
	if [[ "$board" == "p385a_sch_ax115" ]]
	then
		echo $energyaverage | xargs printf "%-10.3f"
		echo $poweraverage | xargs printf "%-10.3f"
	fi
	echo
done

unset CL_CONTEXT_COMPILER_MODE_ALTERA