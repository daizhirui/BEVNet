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
Makefile.ps1
Set-Location ..\..\..\..
pip install -r requirements.txt
conda env update -f environment-win.yml

"Conda environment bevnet is created."
