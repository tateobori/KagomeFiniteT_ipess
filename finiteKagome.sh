#!/bin/bash
#SBATCH -p gr3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1

##### Example of output files
# #SBATCH -J single   # Job name
# #SBATCH -o %x.o%J   # Standard output
# #SBATCH -e %x.e%J   # Standard error

python3 ./src/finite_kagome.py --D 6 --beta_end 1000.0 --dt 0.001 --instate ising-D6-Hx0050 --Hx 0.050
