#!/bin/bash

# be sure to load necessary libraries, such as gcc/4.9+, openmpi, etc.

python lprep.py

mpirun -n 2 python lmain.py
