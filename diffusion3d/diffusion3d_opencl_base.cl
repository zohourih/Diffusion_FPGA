//============================================================================================================
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//
// Using, modifying, and distributing this kernel file is permitted for educational, research, and non-profit
// use cases, as long as this copyright block is kept intact. Using this kernel file in any shape or form,
// including using it as a template/skeleton to develop similar code, is forbidden for commercial/for-profit
// purposes, except with explicit permission from the author (Hamid Reza Zohouri).
//
// Contact point: https://www.linkedin.com/in/hamid-reza-zohouri-9aa00230/
//=============================================================================================================

#include "diffusion3d_opencl_common.h"

#ifndef CSIZE
	#define CSIZE 16
#endif

typedef struct
{
	float data[ASIZE];
} CHAN_WIDTH;

typedef struct
{
	int data[6];
} INT6;

typedef struct
{
	float data[7];
} FLOAT7;

// input shift register parameters
#define IN_SR_BASE		2 * RAD * BLOCK_X * BLOCK_Y			// this shows the point to write into the shift register; RAD rows for top neighbors, one row for current, and (RAD - 1) for bottom
#define IN_SR_SIZE		IN_SR_BASE + ASIZE					// ASIZE indexes are enough for the bottommost row
#define SR_OFF_C		RAD * BLOCK_X * BLOCK_Y				// shift register offset for current cell

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
channel FLOAT7     const_fl_ch[TIME + 1]  __attribute__((depth(0)));
channel INT6       const_int_ch[TIME + 1] __attribute__((depth(0)));

__attribute__((max_global_work_dim(0)))
__kernel void constants(const int             nx_,			// x dimension size
                        const int             ny_,			// y dimension size
                        const int             nz_,			// z dimension size
                        const float           cc_,
                        const float           cw_, 
                        const float           ce_,
                        const float           cs_,
                        const float           cn_,
                        const float           cb_, 
                        const float           ct_,
                        const int             last_col_,		// exit condition for in x direction
                        const int             comp_exit_,		// exit condition for compute loop
                        const int             rem_iter_)		// remaining iterations
{
	// ugly work-around to prevent the stupid compiler from inferring ultra-deep channels
	// for passing the constant values which wastes a lot of Block RAMs.
	// this involves creating a false cycle of channels and passing the values through all
	// the autorun kernels and back to this kernel; this disables the compiler's channel depth optimization.
	FLOAT7 constants1_;
	constants1_.data[0] = ce_;
	constants1_.data[1] = cw_;
	constants1_.data[2] = cn_;
	constants1_.data[3] = cs_;
	constants1_.data[4] = ct_;
	constants1_.data[5] = cb_;
	constants1_.data[6] = cc_;

	INT6 constants2_;
	constants2_.data[0] = nx_;
	constants2_.data[1] = ny_;
	constants2_.data[2] = nz_;
	constants2_.data[3] = last_col_;
	constants2_.data[4] = comp_exit_;
	constants2_.data[5] = rem_iter_;

	write_channel(const_fl_ch[0] , constants1_);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel(const_int_ch[0], constants2_);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const FLOAT7 constants1 = read_channel(const_fl_ch[TIME]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const INT6 constants2 = read_channel(const_int_ch[TIME]);

	const float ce = constants1.data[0];
	const float cw = constants1.data[1];
	const float cn = constants1.data[2];
	const float cs = constants1.data[3];
	const float ct = constants1.data[4];
	const float cb = constants1.data[5];
	const float cc = constants1.data[6];

	const int nx = constants2.data[0];
	const int ny = constants2.data[1];
	const int nz = constants2.data[2];
	const int last_col  = constants2.data[3];
	const int comp_exit = constants2.data[4];
	const int rem_iter  = constants2.data[5];
}

__kernel void read(__global const float* restrict f1,			// input
                            const int             nx,			// x dimension size
                            const int             ny,			// y dimension size
                            const int             pad)			// padding for better memory access alignment
{
	int x = get_local_id(0) * ASIZE;
	int gidx = get_group_id(0);
	int y = get_local_id(1);
	int gidy = get_group_id(1);
	int z = get_global_id(2);
	int bx = gidx * (BLOCK_X - BACK_OFF);					// block offset in x direction
	int by = gidy * (BLOCK_Y - BACK_OFF);					// block offset in y direction
	int gx = bx + x - HALO_SIZE;							// global x position offset, adjusted for halo
	int gy = by + y - HALO_SIZE;							// global y position offset, adjusted for halo

	CHAN_WIDTH in;

	// read data from memory
	#pragma unroll
	for (int i = 0; i < ASIZE; i++)
	{
		int real_x = gx + i;							// global x position
		int index = real_x + gy * nx + z * nx * ny;			// index to read from memory

		// input value
		if (real_x >= 0 && gy >= 0 && real_x < nx && gy < ny)	// avoid out-of-bound indexes in x and y directions; there is also nothing to read on the bottommost row
		{
			in.data[i] = f1[pad + index]; 				// read new values from memory
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

	const FLOAT7 constants1 = read_channel(const_fl_ch[ID]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const INT6  constants2 = read_channel(const_int_ch[ID]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel(const_fl_ch[ID + 1] , constants1);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel(const_int_ch[ID + 1], constants2);

	const float ce = constants1.data[0];
	const float cw = constants1.data[1];
	const float cn = constants1.data[2];
	const float cs = constants1.data[3];
	const float ct = constants1.data[4];
	const float cb = constants1.data[5];
	const float cc = constants1.data[6];

	const int nx = constants2.data[0];
	const int ny = constants2.data[1];
	const int nz = constants2.data[2];
	const int last_col  = constants2.data[3];
	const int comp_exit = constants2.data[4];
	const int rem_iter  = constants2.data[5];

	float in_sr[IN_SR_SIZE];								// for spatial blocking

	// initialize
	#pragma unroll
	for (int i = 0; i < IN_SR_SIZE; i++)
	{
		in_sr[i] = 0.0f;
	}

	// starting points
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;
	int by = 0;
	int index = 0;
	
	while (index != comp_exit)
	{
		index++;

		int comp_offset_z = z - RAD;						// global z position, will be out-of-bound for first and last RAD iterations

		CHAN_WIDTH in, out;
		
		// shift
		#pragma unroll
		for (int i = 0; i < IN_SR_BASE; i++)
		{
			in_sr[i] = in_sr[i + ASIZE];
		}

		// read input values
		if (comp_offset_z < nz - RAD)						// nothing new to read on last RAD planes
		{
			in = read_channel(in_ch[ID]);					// read input values from channel as a vector from previous time step (or read kernel for ID == 0)
		}

		#pragma unroll
		for (int i = 0; i < ASIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;					// global x position, adjusted for halo
			int gy = by + y - HALO_SIZE;					// global y position, adjusted for halo
			int real_x = gx + i;						// global x position

			float north[RAD], south[RAD], east[RAD], west[RAD], below[RAD], above[RAD], current;
			
			in_sr[IN_SR_BASE + i] = in.data[i];			// read input values as array elements

			current = in_sr[SR_OFF_C + i];				// current index

#BORDER_CONDITIONS

			// write output values as array elements
			if (ID < rem_iter)							// if iteration is not out of bound
			{
				out.data[i] = current * cc;
				#pragma unroll
				for (int j = 0; j < RAD; j++)
				{
					out.data[i] = out.data[i]   +
					              west[j]  * cw + east[j]  * ce +
						         south[j] * cs + north[j] * cn +
						         below[j] * cb + above[j] * ct;
				}
			}
			else
			{
				out.data[i] = current;					// pass input data directly to output
			}

			//int comp_offset = real_x + gy * nx + comp_offset_z * nx * ny; // global index, will be out-of-bound for first and last iterations
			//if (comp_offset_z >= 0 && gy >=0 && comp_offset >= 0 && comp_offset < 10)
				//printf("c*cc: %.7E, n*cn: %.7E, s*cs: %.7E, e*ce: %.7E, w*cw: %.7E, a*ca: %.7E, b*cb: %.7E, out: %.7E\n", current * cc, north * cn, south* cs, east * ce, west * cw, above * ct, below * cb, out.data[i]);
				//printf("x: %02d, y: %02d, z: %02d, index: %04d, current: %.7E, north: %.7E,%.7E, south: %.7E.%.7E, west: %.7E,%.7E, east: %.7E,%.7E, above: %.7E,%.7E, below: %.7E,%.7E, out: %.7E\n", real_x, gy, comp_offset_z, comp_offset, current, north[0], north[1], south[0], south[1], west[0], west[1], east[0], east[1], above[0], above[1], below[0], below[1], out.data[i]);
		}

		// write output values
		if (comp_offset_z >= 0)							// not necessary on the first out-of-bound RAD rows
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
			// equivalent to y = (y + 1) % BLOCK_Y
			y = (y + 1) & (BLOCK_Y - 1);					// go to next row

			if (y == 0)
			{
				if (z == nz - 1 + RAD)					// if on last compute plane (compute traverses RAD more planes than memory read/write)
				{
					z = 0;							// reset plane number

					if (bx == last_col)					// border of plane in x direction
					{
						bx = 0;						// reset block column
						by += BLOCK_Y - BACK_OFF;		// go to next block in y direction, account for halos
					}
					else
					{
						bx += BLOCK_X - BACK_OFF;		// go to next block in x direction, account for halos
					}
				}
				else
				{
					z++;								// go to next plane
				}
			}
		}
	}
}

__kernel void write(__global       float* restrict f2,			// input
                             const int             nx,			// x dimension size
                             const int             ny,			// y dimension size
                             const int             pad)		// padding for better memory access alignment
{
	int x = get_local_id(0) * ASIZE;
	int gidx = get_group_id(0);
	int y = get_local_id(1);
	int gidy = get_group_id(1);
	int z = get_global_id(2);
	int bx = gidx * (BLOCK_X - BACK_OFF);					// block offset in x direction
	int by = gidy * (BLOCK_Y - BACK_OFF);					// block offset in y direction
	int gx = bx + x - HALO_SIZE;							// global x position offset, adjusted for halo
	int gy = by + y - HALO_SIZE;							// global y position offset, adjusted for halo
	int real_block_y = y - HALO_SIZE;						// local y position in block, adjusted for halo

	CHAN_WIDTH out;
	out = read_channel(out_ch);							// read output values from channel as a vector

	#pragma unroll
	for (int i = 0; i < ASIZE; i++)
	{
		int real_x = gx + i;							// global x position
		int index = real_x + gy * nx + z * nx * ny;			// index to write to memory
		int real_block_x = x + i - HALO_SIZE;				// local x position in block, adjusted for halo

		// the following condition is to avoid halos and going out of bounds in all axes
		if (real_block_x >= 0 && real_block_y >= 0 && real_block_x < BLOCK_X - 2 * HALO_SIZE && real_block_y < BLOCK_Y - 2 * HALO_SIZE && real_x < nx && gy < ny)
		{
			f2[pad + index] = out.data[i];
		}
	}
}
