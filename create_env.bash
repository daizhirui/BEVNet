#!/bin/bash
set -e  # stop on error
if test -z "$(which conda)"
then
    echo "Please install Anaconda3 or Miniconda at first."
    return
fi

conda create -y --name bevnet python=3.8 pip
# https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
eval "$(conda shell.bash hook)"
conda activate bevnet
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda deactivate
conda activate bevnet
pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd src/models/cspnet/lib || return
pip install Cython
make
cd - || return
pip install -r requirements.txt
conda env update -n bevnet -f environment-linux.yml

echo "Conda environment bevnet is created."
