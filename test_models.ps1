mkdir .\log
mkdir .\log\test

$models = @( `
    "BEVNet-all\mixed",
    "BEVNet-head\mixed",
    "BEVNet-feet\mixed",
    "BEVNet-no_attn\mixed",
    "BEVNet-shared\mixed",
    "CSRNet\CSRNet2BEV-finetune",
    "DSSINet\DSSINet2BEV-finetune",
    "faster_rcnn_r_50_fpn_3x\FasterRCNN2BEV-finetune",
    "mask_rcnn_r_50_fpn_3x\MaskRCNN2BEV",
    "ivnet\CC-Oracle",
    "ivnet\Feet2BEV",
    "ivnet\Head2BEV"
)

foreach ($model in $models) {
    if (-not(Test-Path -Path "log\test\$model\test\model-output.h5" -PathType Leaf)) {
        python src\test.py --task-option-file checkpoints\$model\option.yaml --use-gpus 0
    }

    python src\visualize_model_output.py `
        --model-output-file log\test\$model\test\model-output.h5 -j 8

    python src\run_metrics.py `
        --task-option-file checkpoints\$model\option.yaml `
        --model-output-file log\test\$model\test\model-output.h5 `
        --output-csv log\test\metric_result.csv `
        --use-gpu 0
}
