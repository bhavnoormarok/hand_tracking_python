#!/bin/bash

sync_from_pixel() {
    rsync -avhP pratikm@pixel:/home/pratikm/data/projects/depth_hand_tracking/$1 $2
}

sync_to_pixel() {
    rsync -avhP $1 pratikm@pixel:/home/pratikm/data/projects/depth_hand_tracking/$2
}

sync_from_pixel ./code/hand_model/amano.py ./code/hand_model/
sync_from_pixel ./code/utils ./code/
sync_from_pixel ./output/hand_model ./output/
sync_from_pixel ./data/mano/mano_v1_2 ./data/mano/
sync_from_pixel ./code/registration ./code/
sync_from_pixel ./code/evaluation/kinect ./code/evaluation/
sync_from_pixel ./code/evaluation/metric.py ./code/evaluation/
sync_from_pixel ./data/kinect/pratik ./data/kinect/
sync_from_pixel ./output/kinect/marked_keypoints/pratik ./output/kinect/marked_keypoints/
sync_from_pixel ./create_env.sh .
sync_from_pixel ./setup_dev_env.sh .

