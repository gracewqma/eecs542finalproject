docker run --gpus all --rm -it \
        -v `pwd`:/root \
        diffaugs \
        python /root/cut/train.py --dataroot /root/cut/datasets/horse2zebra_200split --name same_augs_both_200 --CUT_mode CUT