// The first set of functions are based on the "read_sensor" example provided by Bittware and depend on Bittware's headers and libraries.
// The second set of functions are based on the example provided in Section 7.5 of Nallatehc's OpenCL A10 BSP Reference Guide and depend on Nallatech's headers and libraries.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#ifdef LEGACY
	#define EXTENSION "clGetBoardExtensionFunctionAddressAltera"
#else
	#define EXTENSION "clGetBoardExtensionFunctionAddressIntelFPGA"
#endif

#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115
	#include "hil.h"
	#include "bmclib.h"
#elif AOCL_BOARD_p385a_sch_ax115
	#include "aocl_mmd.h"

	#ifdef __APPLE__
		#include <OpenCL/opencl.h>
	#else
		#include <CL/cl.h>
	#endif
#endif

#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115
//====================================================================================================================================
// FPGA Energy Calculator for Bittware's FPGA boards
//====================================================================================================================================

// Bittware's Toolkit has to be installed and correct path to include and library folders from the toolkit have to be provided in make.config
// Example compilation command is as follows:
// gcc -fopenmp test.c -o test -L/opt/bwtk/lib -I/opt/bwtk/include -I/opt/bwtk/include/resources -lbwhil -lbmclib -DLINUX

// This function works very similar to the GPU power function
// Returns average power usage in Watt from when it is called until when "flag" becomes one
// Sampling is done every 10 milliseconds
// The host code should have two OpenMP threads, one running the OpenCL kernel and the other calling this function
// A "#pragma omp barrier" should be put before the kernel call
// Flag should become one in the kernel thread after kernel execution has finished (after clFinish())

#define DeviceNum 0 // Used to choose target FPGA board
#define SDR       0 // Used to choose the target sensor, sensor 0 is the power sensor on the AL10P4 board

BMC_Handle bmc;
HHil hil;
HDevice hdev;

static inline void cleanup()
{
	if(bmc)
		bmc_disconnect(bmc);
	if(hdev)
		hil_close(hdev);
	if(hil)
		hil_exit(hil);

	#pragma omp barrier
}

static inline double GetPowerFPGA(int* flag)
{
	int boardtype = 0, sdr_ok = 0, device, sdr, ret, i;
	BmcPeripheralTable* p_periph_table = NULL;
	char value[256];
	char state[256];
	U8 record[80];

	bmc = NULL;
	hil = NULL;
	hdev = NULL;

	device = DeviceNum;
	sdr = SDR;

	unsigned int power;
	size_t count = 0, powerSum = 0;

	if(!(hil = hil_init(HILINIT_NO_OPTION)))
	{
		printf("Failed to initialize Host Interface Library.\n");
		cleanup();
		return -1;
	}

	if(!(hdev = hil_open(hil, device, HILOPEN_NO_OPTION)))
	{
		HilError hilerr = hil_get_last_error(hil);
		if((hilerr) == HIL_BUSY)
		{
			printf("Device is busy. Please retry after closing any BittWare applications using the device.\n");
		}
		else
		{
			printf("Could not open device due to error: %s.\n", hil_get_error_string(hilerr));
		}
		cleanup();
		return -1;
	}

	if(!(bmc = bmc_connect_device(hdev)))
	{
		printf("Could not connect to BMC on device.\n");
		cleanup();
		return -1;
	}

	hil_get_device_value(hdev, HIL_BOARD_TYPE, &boardtype);
	p_periph_table = (BmcPeripheralTable*)bmclib_get_peripheral_table((HilBoardType)boardtype);

	if (!p_periph_table)
	{
		printf("Unsupported board type %d.\n", boardtype);
		cleanup();
		return -1;
	}

	for(i=0; p_periph_table->entries[i].name; i++)
	{
		if((p_periph_table->entries[i].type == ComponentTypeSensor) && (p_periph_table->entries[i].sdr == sdr))
		{
			int next_sdr = sdr;
			if((ret = bmc_sdr_get_record(bmc, &next_sdr, record, sizeof(record))) < 0)
			{
				printf("Unable to get SDR %d.\n", sdr);
				cleanup();
				return -1;
			}
			sdr_ok = 1;

			#pragma omp barrier
			while(*flag == 0)
			{
				// Returns device power usage in Watt
				bmc_sdr_read_sensor(bmc, record, value, sizeof(value), state, sizeof(state));

				power = atof(value);
				powerSum = powerSum + power;
				count++;

				// Sleep for 10 ms
				usleep(10000);
			}
			break;
		}
	}

	if (!sdr_ok)
	{
		printf("Invalid SDR index %d.\n", sdr);
		cleanup();
		return -1;
	}

	return (double)(powerSum)/(double)(count);
}

#elif AOCL_BOARD_p385a_sch_ax115
//====================================================================================================================================
// FPGA Energy Calculator for Nallatech's FPGA boards
//====================================================================================================================================

// AOCL_BOARD_PACKAGE_ROOT must point to Nallatech's BSP that includes the aocl_mmd.h header file.
// Unlike the function for the Bittware board, the function for the Nallatech's board needs the OpenCL device list as input

// This function works very similar to the GPU power function
// Returns average power usage in Watt from when it is called until when "flag" becomes one
// Sampling is done every 10 milliseconds
// The host code should have two OpenMP threads, one running the OpenCL kernel and the other calling this function
// A "#pragma omp barrier" should be put before the kernel call
// Flag should become one in the kernel thread after kernel execution has finished (after clFinish())

typedef void* (*get_board_extension_function_address_fn_t)(const char* func_name, cl_device_id device);
typedef void* (*aocl_mmd_card_info_fn_t)(const char*, aocl_mmd_info_t, size_t, void*, size_t* );

static inline void cleanup()
{
	#pragma omp barrier
}

static inline double GetPowerFPGA(int* flag, cl_device_id* device)
{
	void *tempPointer;

	float power, powerSum = 0;
	size_t count = 0;
	size_t returnedSize;

	get_board_extension_function_address_fn_t board_extension_function_address = (get_board_extension_function_address_fn_t) clGetExtensionFunctionAddress (EXTENSION);
	if (board_extension_function_address == NULL)
	{
		printf ("Failed to get %s.\n", EXTENSION);
		cleanup();
		return -1;
	}
	
	tempPointer = board_extension_function_address("aocl_mmd_card_info", device[0]);
	
	aocl_mmd_card_info_fn_t aocl_mmd_card_info_fn = (aocl_mmd_card_info_fn_t)tempPointer;
	if (aocl_mmd_card_info_fn == NULL)
	{
		printf ("Failed to get aocl_mmd_card_info_fn address.\n");
		cleanup();
		return -1;
	}

	#pragma omp barrier
	while(*flag == 0)
	{
		// Returns device power usage in Watt
		// aclnalla_pcie0 is the board name string
		aocl_mmd_card_info_fn("aclnalla_pcie0", AOCL_MMD_POWER, sizeof(float), (void*) &power, &returnedSize);

		if (power >= 10.0 && power <= 80.0)
		{
			powerSum = powerSum + power;
			count++;
		}

		// Sleep for 10 ms
		usleep(10000);
	}

	return (double)(powerSum)/(double)(count);
}

#endif

// Returns amount of energy used in jouls
// "power" is average power usage in Watt from the GetPowerGPU() function
// "time" is run time in ms from one of our time measurement helper functions
static inline double GetEnergyFPGA(double power, double time)
{
	return (power * time); // Time is in seconds and power is in Watt
}
