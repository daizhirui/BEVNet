$ErrorActionPreference='stop'
if (-not(Get-Command conda)) {
    "Please install Anaconda3 or Miniconda at first."
    return
}

conda create -y --name bevnet python=3.8 pip
conda activate bevnet
conda install -y pywin32
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda deactivate
conda activate bevnet
pip install 'git+https://github.com/facebookresearch/detectron2.git'
Set-Location src\models\cspnet\lib_win32
pip install Cython
Makefile.ps1
Set-Location ..\..\..\..
pip install -r requirements.txt
conda env update -n bevnet -f environment-win.yml
# https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp
pip uninstall -y numpy
pip install -y numpy

"Conda environment bevnet is created."
