python SuperGAT/_tokengt_main.py \
    --dataset-class MyCoauthor \
    --dataset-name Physics \
    --model-name TokenGT \
    --encoder-layers 2 \
    --encoder-embed-dim 192 \
    --encoder-ffn-embed-dim 192 \
    --encoder-attention-heads 2 \
    --performer-generalized-attention \
    --performer-feature-redraw-interval 99 \
    --input-dropout 0.1 \
    --activation-dropout 0.1 \
    --dropout 0.1 \
    --classifier-dropout 0.5 \
    --lap-pe \
    --lap-pe-k 64 \
    --lap-pe-eig-dropout 0.1 \
    --broadcast-features
