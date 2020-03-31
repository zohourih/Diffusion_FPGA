# Define the following variables for the INTEL compiler:
# CC
# CXX
# FC
# DEBUG_FLAGS
# OPT_FLAGS
# WARNING_FLAGS
# OPENMP_CFLAGS
# OPENMP_LDFLAGS

CC = icc
CXX = icpc
FC = ifort

ifneq ($(DEBUG),)
DEBUG_FLAGS = -g
OPT_FLAGS =
else
DEBUG_FLAGS =
OPT_FLAGS = -O3 -xHost 
endif

WARNING_FLAGS = -Wall

OPENMP_CFLAGS = -qopenmp
OPENMP_LDFLAGS = -qopenmp

CXX_LIBS = 
F90_LIBS = 
