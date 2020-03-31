// Radius of stencil, e.g 5-point stencil => 1
#ifndef RAD
  #define RAD  1
#endif

// Block size
#ifndef BLOCK_X
  #define BLOCK_X 4096
#endif

// Memory access size
#ifndef ASIZE
  #ifdef AOCL_BOARD_de5net_a7
    #define ASIZE  8
  #elif defined (AOCL_BOARD_a10pl4_dd4gb_gx115) || defined (AOCL_BOARD_p385a_sch_ax115)
    #define ASIZE  16
  #endif
#endif

// Radius of stencil, e.g 5-point stencil => 1
#ifndef RAD
  #define RAD  1
#endif

// Number of parallel time steps
#ifndef TIME
  #ifdef AOCL_BOARD_de5net_a7
    #define TIME 4
  #elif defined (AOCL_BOARD_a10pl4_dd4gb_gx115) || defined (AOCL_BOARD_p385a_sch_ax115)
    #define TIME 12
  #endif
#endif

// Padding to fix alignment for time steps that are not a multiple of 8
#ifndef PAD
  #define PAD (TIME * RAD) % 16
#endif

#define HALO_SIZE  TIME * RAD        // Halo size
#define BACK_OFF  2 * HALO_SIZE      // Back-off for going to next block