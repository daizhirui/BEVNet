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
conda install -y pytorch torchvision cudatoolkit=11.1 numpy=1.20.3 -c pytorch-lts -c nvidia
conda deactivate
conda activate bevnet
python -m pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
cd src/models/cspnet/lib || return
pip install Cython
make clean
make
cd - || return
pip install -r requirements.txt
conda env update -n bevnet -f environment-linux.yml

echo "Conda environment bevnet is created."
