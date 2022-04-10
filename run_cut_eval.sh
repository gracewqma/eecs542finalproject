docker run --gpus all --rm -it \
        -v `pwd`:/root \
        diffaugs \
        python /root/cut/test.py --dataroot /root/cut/datasets/horse2zebra --name same_augs_both --CUT_mode CUT