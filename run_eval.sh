docker run --gpus all --rm -it \
        -v `pwd`:/root \
        diffaugs \
        python /root/eval.py --exp_name dcgan_color_only