CUDA_VISIBLE_DEVICES=0 \
python3  main.py \
        --train_dataset ic15_train \
        --val_dataset ic15_val \
        --max_length 25 \
        --data_root '/path/data/' \
        --batch_size 1 \
        --max_size_test 1920 \
        --min_size_test 1280 \
        --depths 6 \
        --lr 0.00001 \
        --pre_norm \
        --num_workers 2 \
        --eval \
        --resume '/path/resume/' \
        --output_dir '/path/save/' \
        --padding_bins 0 \
        --train_point point \
        --pad_rec 