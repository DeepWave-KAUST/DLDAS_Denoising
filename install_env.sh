#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh
# 

echo 'Creating Package environment'

# create conda env
conda env create -f DASDL.yml
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh

conda activate DASDL
#conda env list
echo 'Created and activated environment:' $(which python)

# check torch works as expected
#echo 'Checking torch version and running a command...'
#:python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'

