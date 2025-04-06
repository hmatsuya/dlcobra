#!/bin/bash
pipenv run python plaintext_to_hcpe.py /mnt/nvme1n1p2/data /mnt/nvme1n1p2/data 30
rm /mnt/nvme1n1p2/data/concatenated.h5
./hcpe_to_hdf5.sh