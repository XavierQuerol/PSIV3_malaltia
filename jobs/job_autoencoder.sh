#!/bin/bash
#SBATCH -n 4 # Number of CPU cores. Maximum 10
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /tmp # Working directory
#SBATCH -t 2-00:05 # Runtime in D-HH:MM
#SBATCH -p tfg # Partition to submit to
#SBATCH --mem 2048 # RAM memory. 2GB solicitados. Maximum 60GB
#SBATCH -o /fhome/mapsiv01/PSIV3_malaltia/autoencoder/file.out # File to which STDOUT will be written
#SBATCH -e /fhome/mapsiv01/PSIV3_malaltia/autoencoder/file.err # File to which STDERR will be written
#SBATCH --gres gpu:1 # Number of requested GPU (1). Max 8
sleep 1
# File .py  to execute
python /fhome/mapsiv01/PSIV3_malaltia/autoencoder/main_autoencoder.py

