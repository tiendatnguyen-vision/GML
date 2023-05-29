python SuperGAT/_tokengt_main.py \
    --dataset-class MyCitationFull \
    --dataset-name CoraFull \
    --model-name TokenGT \
    --encoder-layers 3 \
    --encoder-embed-dim 256 \
    --encoder-ffn-embed-dim 256 \
    --encoder-attention-heads 2 \
    --performer-generalized-attention \
    --performer-feature-redraw-interval 99 \
    --input-dropout 0.0 \
    --activation-dropout 0.0 \
    --dropout 0.1 \
    --classifier-dropout 0.0 \
    --lap-pe \
    --lap-pe-k 60 \
    --lap-pe-eig-dropout 0.1 \
    --broadcast-features \
    --drop-edge-tokens \
    --num-total-runs 5 \