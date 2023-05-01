#!/bin/bash
# module load radmc3d
python3 src/main.py
cd output
radmc3d mctherm setthreads 12
radmc3d image imolspec 1 iline 2 vkms 1.0 incl 90 phi 0
python3 ../src/show_r3d.py  