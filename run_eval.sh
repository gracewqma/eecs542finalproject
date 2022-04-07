docker run --gpus all --rm -it \
        -v `pwd`:/root \
        neuralsceneflow \
        python /root/eval.py --exp_name dcgan_no_aug