#!/bin/bash

rad=$(($1 - 1))
line_size=70
shift_reg_name=in_sr
neighbors=(west east south north)
sign=("-" "+" "-" "+")
multiplier=("" "" " * BLOCK_X" " * BLOCK_X")
left_hand=(real_x real_x comp_offset_y comp_offset_y)
right_hand=(" " " (nx - 1) - " " " " (ny - 1) - ")

for k in $(seq 0 $((${#neighbors[@]} - 1)))
do
	echo -en "\t\t\t// ${neighbors[k]} neighbor"; echo ""

	echo -en "\t\t\tif (${left_hand[k]} ==${right_hand[k]}0)"; echo ""
	echo -en "\t\t\t{"; echo ""
	for j in $(seq 0 $rad)
	do
		echo -en "\t\t\t\t${neighbors[k]}[$j] = current;"; echo ""
	done
	echo -en "\t\t\t}"; echo ""

	for i in $(seq 1 $rad)
	do
		echo -en "\t\t\telse if (${left_hand[k]} ==${right_hand[k]}$i)"; echo ""
		echo -en "\t\t\t{"; echo ""
		for j in $(seq 0 $rad)
		do
			if [[ j -lt i ]]
			then
				offset=$(( j + 1 ))
				echo -en "\t\t\t\t${neighbors[k]}[$j] = $shift_reg_name[SR_OFF_C ${sign[k]} $offset${multiplier[k]} + i];"; echo ""
			else
				offset=$(( i - 1 ))
				echo -en "\t\t\t\t${neighbors[k]}[$j] = ${neighbors[k]}[$offset];"; echo ""
			fi
		done
		echo -en "\t\t\t}"; echo ""
	done

	echo -en "\t\t\telse"; echo ""
	echo -en "\t\t\t{"; echo ""
	for j in $(seq 0 $rad)
	do
			offset=$(( j + 1 ))
			echo -en "\t\t\t\t${neighbors[k]}[$j] = $shift_reg_name[SR_OFF_C ${sign[k]} $offset${multiplier[k]} + i];"; echo ""
	done
	echo -en "\t\t\t}"; echo ""

	if [[ k -ne $((${#neighbors[@]} - 1)) ]]
	then
		echo ""
	fi
done
