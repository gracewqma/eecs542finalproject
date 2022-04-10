docker run --gpus all --rm -it \
        -v `pwd`:/root \
        diffaugs \
        python /root/split_datasets.py