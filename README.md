BEV-Net: Assessing Social Distancing Compliance by Joint People Localization and Geometric Reasoning
======

PyTorch implementation of ICCV2021 paper, *BEV-Net: Assessing Social
Distancing Compliance by Joint People Localization and Geometric Reasoning*,
for estimating camera pose and analysing social distancing compliance with
geometric reasoning:

![](assets/teaser.png)

# Setup

## Prerequisites

- Windows or Linux
- NVIDIA GPU + CUDA cuDNN

### Tested Environments

- Windows
    - PyTorch 1.8.2
    - CUDA 11.1
- Linux
    - PyTorch 1.8.2
    - CUDA 11.1

## Create Environment

### Linux

```bash
# make sure replace the path with the correct one
export CUDAHOME="/usr/local/cuda"
bash create_env.bash
```

### Windows

```ps1
Set-ExecutionPolicy unrestricted
create_env.ps1
```

## Download Data

```shell
git submodule update --init --recursive ./data
```

## Prepare Dataset (Optional)

Dataset should be ready when the submodule at `data` is pulled.

```shell
python src/datasets/cityuhk/build_dataset.py
python src/datasets/cityuhk/build_datalist.py
```

## Download Checkpoints (Optional)

We provide all the checkpoints of models we used, including the baselines.
```shell
git submodule update --init --recursive ./checkpoints_tar_parts
```

## Uncompress Checkpoints (Optional)

- Linux
```shell
bash uncompress_checkpoints.bash
```
- Windows: you may need to use tools like `7zip` to uncompress the files.

We also provide the bash script to compress the checkpoints again. So, you
can delete `checkpoints_tar_parts` if you like to.

# How to use

Please make sure the environment is activated

```bash
conda activate bevnet
```

## Train Models

```shell
python ./src/train.py \
    --task-option-file ./configs/bevnet/mixed-all.yaml --use-gpus 0
```

## Test Models

- If you want to collect all the test results in a single folder, please make sure `log/test` is created before running any test. Otherwise, test results will be saved along with the checkpoint file.

- Generate model output for the test dataset and calculate losses
```shell
python src/test.py \
    --task-option-file checkpoints/BEVNet-all/mixed/option.yaml --use-gpus 0
```
- Generate visualization of the model output
```shell
python src/visualize_model_output.py \
    --model-output-file log/test/BEVNet-all/mixed/test/model-output.h5 -j 8
```
- Run the SDCA metrics
```shell
python src/run_metrics.py \
        --task-option-file checkpoints/BEVNet-all/mixed/option.yaml \
        --model-output-file log/test/BEVNet-all/mixed/test/model-output.h5 \
        --output-csv log/test/metric_result.csv \
        --use-gpu 0
```

**To test all the provided models, run the script**:
- Linux
```shell
bash test_models.bash
```
- Windows
```shell
test_models.ps1
```

# Citation

```latex
@misc{dai2021bevnet,
      title={BEV-Net: Assessing Social Distancing Compliance by Joint People Localization and Geometric Reasoning},
      author={Zhirui Dai and Yuepeng Jiang and Yi Li and Bo Liu and Antoni B. Chan and Nuno Vasconcelos},
      year={2021},
      eprint={2110.04931},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
