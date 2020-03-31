#!/usr/bin/env python

import subprocess
import sys

max_threads = 1024
min_threads = 128

bx=[32, 64, 128, 256]
by=[1, 2, 4, 8, 16, 32]
#gz=[1, 2, 4, 8, 16, 32, 64, 128]
gz=[1, 2, 4, 8, 16, 32, 64]
unroll=[0]
    
all_configs = []
for x in bx:
    for y in by:
        for z in gz:
            for u in unroll:            
                c = (x, y, z, u)
                num_threads = x*y
                if num_threads > max_threads:
                    continue
                if num_threads < min_threads:
                    continue
                all_configs.append(c)

make_opt = " ".join(sys.argv[1:])
print make_opt
print "debug " + ", ".join(sys.argv)
for c in all_configs:
    print c
    proc = subprocess.Popen("make clean", shell=True)
    proc.wait()
    command="make -k -j10 cuda BLOCK_X=%d BLOCK_Y=%d GRID_Z=%d UNROLL=%d %s" % \
        (c[0], c[1], c[2], c[3], make_opt)
    print command
    proc = subprocess.Popen(command, shell=True)
    proc.wait()
    dirname = "%dx%dx%d_%d" % (c[0], c[1], c[2], c[3])
    command = "mkdir " + dirname
    proc = subprocess.Popen(command, shell=True)
    proc.wait()
    command = "mv *.exe " + dirname
    proc = subprocess.Popen(command, shell=True)
    proc.wait()
    
    

