if test -z "$(which conda)"
then
    echo "Please install Anaconda3 or Miniconda at first."
    return
fi

conda create -y --name bevnet python=3.8 pip
conda activate bevnet
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd src/models/cspnet/lib || return
make
cd - || return
conda env update -f environment.yml

echo "Conda environment bevnet is created."
