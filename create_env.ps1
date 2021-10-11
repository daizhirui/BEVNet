if (-not(Get-Command conda)) {
    "Please install Anaconda3 or Miniconda at first."
    return
}

conda create -y --name bevnet python=3.8 pip
conda activate bevnet
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd src\models\cspnet\lib_win32
make
cd ..\..\..\..
conda env update -f environment.yml

"Conda environment bevnet is created."
