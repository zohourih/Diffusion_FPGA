# Define the following variables for the PGI compiler:
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
# ACC_CFLAGS
# ACC_CXXFLAGS
# ACC_FFLAGS
# ACC_LDFLAGS

CC = pgcc
CXX = pgc++
FC = pgfortran

ifneq ($(DEBUG),)
DEBUG_FLAGS = -g
OPT_FLAGS =
else
DEBUG_FLAGS =
OPT_FLAGS = -O3 -fastsse
endif

WARNING_FLAGS =

OPENMP_CFLAGS = -mp
OPENMP_LDFLAGS = -mp=allcores

CXX_LIBS = -pgcpplibs
F90_LIBS = -pgf90libs

ACC_COMMON_FLAGS = -acc -ta=nvidia -Minfo=accel
ACC_CFLAGS = $(ACC_COMMON_FLAGS)
ACC_CXXFLAGS = $(ACC_COMMON_FLAGS)
ACC_FFLAGS = $(ACC_COMMON_FLAGS)

ACC_LDFLAGS = -acc -ta=nvidia -Minfo=accel
