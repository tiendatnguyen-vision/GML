python SuperGAT/_tokengt_main.py \
    --dataset-class MyCitationFull \
    --dataset-name Cora_ML \
    --model-name TokenGT \
    --encoder-layers 3 \
    --encoder-embed-dim 128 \
    --encoder-ffn-embed-dim 128 \
    --encoder-attention-heads 2 \
    --performer-generalized-attention \
    --performer-feature-redraw-interval 99 \
    --input-dropout 0.3 \
    --activation-dropout 0.1 \
    --dropout 0.1 \
    --classifier-dropout 0.5 \
    --lap-pe \
    --lap-pe-k 64 \
    --lap-pe-eig-dropout 0.1 \
    --broadcast-features \
    --drop-edge-tokens \
    --use-kd \
    --w-kd-feat 0.4 \
    --w-kd-response 0.4 \
    --teacher-name GCN \
    --teacher-lr 0.001 \
    --teacher-l2-lambda 0.01 \
    --teacher-heads 8 \
    --teacher-dropout 0.6 \
    --teacher-num-hidden-features 128 \
    --student-intermediate-index 2 \
    --num-total-runs 3 \
