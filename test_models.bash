set -e
mkdir -p log/test
for i in 'BEVNet-all/mixed' 'BEVNet-head/mixed' 'BEVNet-feet/mixed' \
    'BEVNet-no_attn/mixed' 'BEVNet-shared/mixed' 'CSRNet/CSRNet2BEV-finetune' \
    'CSPNet/CSPNet2BEV' 'DSSINet/DSSINet2BEV-finetune' \
    'faster_rcnn_r_50_fpn_3x/FasterRCNN2BEV-finetune' \
    'mask_rcnn_r_50_fpn_3x/MaskRCNN2BEV' 'ivnet/CC-Oracle' 'ivnet/Feet2BEV' \
    'ivnet/Head2BEV' ; do

    model_output_file="log/test/$i/test/model-output.h5"

    if ! test -f $model_output_file ; then
        python src/test.py \
            --task-option-file checkpoints/$i/option.yaml \
            --use-gpus 0
    fi

    python src/visualize_model_output.py \
        --model-output-file log/test/$i/test/model-output.h5 -j 8

    python src/run_metrics.py \
        --task-option-file checkpoints/$i/option.yaml \
        --model-output-file log/test/$i/test/model-output.h5 \
        --output-csv log/test/metric_result.csv \
        --use-gpu 0
done
