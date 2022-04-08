docker run --gpus all --rm -it \
        -v `pwd`:/root \
        diffaugs \
        python train.py --dataroot datasets/horse2zebra --name test --CUT_mode CUT