python main.py --data_dir="../IXI_2d" \
               --loss="1*L1" \
               --save="firefly" \
               --modal1="T1" \
               --modal2="DTI" \
               --model="test1" \
               --train_ratio=0.8 \
               --val_ratio=0.05 \
               --test_ratio=0.15 \
               --epochs=200 \
               --batch_size=4 \
               --lr=5e-4 \
               --pre_train=None \
               --n_GPUs=1 \
               --gpu_ids 0 \
               --weight_decay=5e-4
