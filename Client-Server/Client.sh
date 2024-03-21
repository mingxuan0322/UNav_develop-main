#!/bin/bash

work_path=/home/endeleze/Desktop/UNav_develop/


host_id=128.122.136.119
port_id=30001
camera_index=0
capture_interval=3

brightness=20
contrast=20
saturation=68
speak_rate=140
volume=2.0
voices=12
script=$work_path/Client-Server/Client.py

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pycolmap

python $script --host_id $host_id --port_id $port_id --camera_index $camera_index --capture_interval $capture_interval \
--brightness $brightness --contrast $contrast --saturation $saturation \
--speak_rate $speak_rate --volume $volume --voices $voices
