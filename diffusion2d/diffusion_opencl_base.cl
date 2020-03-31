//============================================================================================================
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//
// Using, modifying, and distributing this kernel file is permitted for educational, research, and non-profit
// use cases, as long as this copyright block is kept intact. Using this kernel file in any shape or form,
// including using it a template/skeleton to develop similar code, is forbidden for commercial/for-profit
// purposes, except with explicit permission from the author (Hamid Reza Zohouri).
//
// Contact point: https://www.linkedin.com/in/hamid-reza-zohouri-9aa00230/
//=============================================================================================================

#include "diffusion_opencl_common.h"

#ifndef CSIZE
	#define CSIZE 16
#endif

typedef struct
{
	float data[ASIZE];
} CHAN_WIDTH;

typedef struct
{
	float data[5];
} FLOAT5;

// input shift register parameters
#define IN_SR_BASE		2 * RAD * BLOCK_X					// this shows the point to write into the shift register; RAD rows for top neighbors, one row for current, and (RAD - 1) for bottom
#define IN_SR_SIZE		IN_SR_BASE + ASIZE					// ASIZE indexes are enough for the bottommost row

// offsets for reading from the input shift register
#define SR_OFF_C		RAD * BLOCK_X						// current

#ifdef LEGACY
	#pragma OPENCL EXTENSION cl_altera_channels : enable
	#define read_channel read_channel_altera
	#define write_channel write_channel_altera
#else
	#pragma OPENCL EXTENSION cl_intel_channels : enable
	#define read_channel read_channel_intel
	#define write_channel write_channel_intel
#endif

channel CHAN_WIDTH in_ch[TIME]            __attribute__((depth(CSIZE)));
channel CHAN_WIDTH out_ch                 __attribute__((depth(CSIZE)));
channel FLOAT5     const_fl_ch[TIME + 1]  __attribute__((depth(0)));
channel int4       const_int_ch[TIME + 1] __attribute__((depth(0)));

__attribute__((max_global_work_dim(0)))
__kernel void constants(const int             nx_,			// x dimension size
                        const int             ny_,			// y dimension size
                        const float           cc_,
                        const float           cw_, 
                        const float           ce_,
                        const float           cs_,
                        const float           cn_,
                        const int             comp_exit_,		// exit condition for compute loop
                        const int             rem_iter_)		// remaining iterations
{
	// ugly work-around to prevent the stupid compiler from inferring ultra-deep channels
	// for passing the constant values which wastes a lot of Block RAMs.
	// this involves creating a false cycle of channels and passing the values through all
	// the autorun kernels and back to this kernel; this disables the compiler's channel depth optimization.
	FLOAT5 constants1_;
	constants1_.data[0] = ce_;
	constants1_.data[1] = cw_;
	constants1_.data[2] = cn_;
	constants1_.data[3] = cs_;
	constants1_.data[4] = cc_;

	int4 constants2_ = (int4)(nx_, ny_, comp_exit_, rem_iter_);

	write_channel(const_fl_ch[0] , constants1_);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel(const_int_ch[0], constants2_);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const FLOAT5 constants1 = read_channel(const_fl_ch[TIME]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const int4 constants2 = read_channel(const_int_ch[TIME]);

	const float ce = constants1.data[0];
	const float cw = constants1.data[1];
	const float cn = constants1.data[2];
	const float cs = constants1.data[3];
	const float cc = constants1.data[4];

	const int nx = constants2.s0;
	const int ny = constants2.s1;
	const int comp_exit = constants2.s2;
	const int rem_iter  = constants2.s3;
}

__kernel void read(__global const float* restrict f1,			// input
                            const int             nx,			// x dimension size
                            const int             pad)			// padding for better memory access alignment
{
	int x = get_local_id(0) * ASIZE;
	int gid = get_group_id(0);
	int y = get_global_id(1);
	int bx = gid * (BLOCK_X - BACK_OFF);					// block offset
	int gx = bx + x - HALO_SIZE;							// global x position offset, adjusted for halo
	CHAN_WIDTH in;

	// read data from memory
	#pragma unroll
	for (int i = 0; i < ASIZE; i++)
	{
		int real_x = gx + i;							// global x position in vector
		int index = y * nx + real_x;						// index to read from memory

		// input values
		if (real_x >= 0 && real_x < nx)					// avoid out-of-bound indexes in x direction
		{
			in.data[i] = f1[pad + index];					// read new values from memory
		}
	}

	write_channel(in_ch[0], in);							// write input values to channel as a vector
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(TIME,1,1)))
__kernel void compute()
{
	const int ID = get_compute_id(0);

	const FLOAT5 constants1 = read_channel(const_fl_ch[ID]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const int4  constants2 = read_channel(const_int_ch[ID]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel(const_fl_ch[ID + 1] , constants1);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel(const_int_ch[ID + 1], constants2);

	const float ce = constants1.data[0];
	const float cw = constants1.data[1];
	const float cn = constants1.data[2];
	const float cs = constants1.data[3];
	const float cc = constants1.data[4];

	const int nx = constants2.s0;
	const int ny = constants2.s1;
	const int comp_exit = constants2.s2;
	const int rem_iter  = constants2.s3;

	float in_sr[IN_SR_SIZE];								// for spatial blocking

	// initialize
	#pragma unroll
	for (int i = 0; i < IN_SR_SIZE; i++)
	{
		in_sr[i] = 0.0f;
	}

	// starting point
	int x = 0;
	int y = 0;
	int bx = 0;
	int index = 0;

	while (index != comp_exit)
	{
		index++;

		int comp_offset_y = y - RAD;						// global y position, will be out-of-bound for first and last iterations

		CHAN_WIDTH in, out;
		
		// shift
		#pragma unroll
		for (int i = 0; i < IN_SR_BASE; i++)
		{
			in_sr[i] = in_sr[i + ASIZE];
		}

		// read input values
		if (comp_offset_y < ny - RAD)						// nothing to read on last row
		{
			in = read_channel(in_ch[ID]);					// read input values from channel as a vector from previous time step (or read kernel for ID == 0)
		}

		#pragma unroll
		for (int i = 0; i < ASIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;					// global x position offset
			int real_x = gx + i;						// global x position

			float north[RAD], south[RAD], east[RAD], west[RAD], current;
			
			in_sr[IN_SR_BASE + i] = in.data[i];			// read input values as array elements

			current = in_sr[SR_OFF_C + i];				// current index

#BORDER_CONDITIONS

			// write output values as array elements
			if (ID < rem_iter)							// if iteration is not out of bound
			{
				out.data[i] = cc * current;
				#pragma unroll
				for (int j = 0; j < RAD; j++)
				{
					out.data[i] = out.data[i]   +
					              cw * west[j]  + ce * east[j] +
						         cs * south[j] + cn * north[j];
				}
			}
			else
			{
				out.data[i] = current;					// pass input data directly to output
			}
			//if (ID == 1 && comp_offset_y >=0 && comp_offset_y < ny && real_x >=0 && real_x < nx)
				//printf("ID: %d, row: %04d, col: %04d, current: %f, left: %f, right: %f, out: %f\n", ID, comp_offset_y, real_x, current, north, south, west, east, out.data[i]);
		}

		// write output values
		if (comp_offset_y >= 0)							// nothing to write on first row
		{
			if (ID == TIME - 1)							// only if last time step
			{
				write_channel(out_ch, out);				// write output values to channel as a vector for write back to memory
			}
			else										// avoid creating the following channel if the next time step "doesn't exist"
			{
				write_channel(in_ch[ID + 1], out);			// write output values to channel as a vector for next time step
			}
		}

		// equivalent to x = (x + ASIZE) % BLOCK_X
		x = (x + ASIZE) & (BLOCK_X - 1);					// move one chunk forward and reset to zero if end of block was reached

		if (x == 0)									// if one block finished
		{
			if (y == ny - 1 + RAD)						// if on last row (compute traverses RAD more rows than memory read/write)
			{
				y = 0;								// reset row number
				bx += BLOCK_X - BACK_OFF;				// go to next block, account for halos
			}
			else
			{
				y++;									// go to next row
			}
		}
	}
}

__kernel void write(__global       float* restrict f2,			// temperature output
                             const int             nx,			// number of columns
                             const int             pad)		// padding for better memory access alignment
{
	int x = get_local_id(0) * ASIZE;
	int gid = get_group_id(0);
	int y = get_global_id(1);
	int bx = gid * (BLOCK_X - BACK_OFF);					// block offset
	int gx = bx + x - HALO_SIZE;							// global x position offset, adjusted for halo
	CHAN_WIDTH out;

	out = read_channel(out_ch);							// read output values from channel as a vector

	#pragma unroll
	for (int i = 0; i < ASIZE; i++)
	{
		int real_x = gx + i;							// global x position in vector
		int real_block_x = x + i - HALO_SIZE;				// local x position in block, adjusted for halo
		int index = y * nx + real_x;						// index to read from memory

		// the following condition is to avoid halos and going out of bounds in either axes
		if (real_block_x >= 0 && real_block_x < BLOCK_X - 2 * HALO_SIZE && real_x < nx)
		{
			f2[pad + index] = out.data[i];
		}
	}
}
