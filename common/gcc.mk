# Define the following variables for the GCC compiler:
# CC
# CXX
# FC
# DEBUG_FLAGS
# OPT_FLAGS
# WARNING_FLAGS
# OPENMP_CFLAGS
# OPENMP_LDFLAGS
# CXX_LIBS
# F90_LIBS

CC = gcc
CXX = g++
FC = gfortran

ifneq ($(DEBUG),)
DEBUG_FLAGS = -g
OPT_FLAGS =
else
DEBUG_FLAGS = 
OPT_FLAGS = -O3
endif

WARNING_FLAGS = -Wall

OPENMP_CFLAGS = -fopenmp
OPENMP_LDFLAGS = -fopenmp

CXX_LIBS = 
F90_LIBS = 
