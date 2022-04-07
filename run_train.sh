docker run --gpus all --rm -it \
        -v `pwd`:/root \
        diffaugs \
        python /root/train.py --config /root/configs/dcgan_translation_only.txt