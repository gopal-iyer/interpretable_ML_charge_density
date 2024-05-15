#!/bin/bash

#SBATCH --nodes=10
#SBATCH --partition=pdebug
#SBATCH --time=01:00:00
#SBATCH --job-name=test_Al
srun -n512 main sample.in >log

