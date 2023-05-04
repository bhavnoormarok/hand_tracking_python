#!/bin/bash

color_1="./data/kinect/pratik/1/color_raw/video.mp4"
depth_1="./data/kinect/pratik/1/depth_raw_cm/video.mp4"

color_2="./data/kinect/pratik/2/color_raw/video.mp4"
depth_2="./data/kinect/pratik/2/depth_raw_cm/video.mp4"

color_3="./data/kinect/pratik/3/color_raw/video.mp4"
depth_3="./data/kinect/pratik/3/depth_raw_cm/video.mp4"

color_4="./data/kinect/pratik/4/color_raw/video.mp4"
depth_4="./data/kinect/pratik/4/depth_raw_cm/video.mp4"

color_5="./data/kinect/pratik/5/color_raw/video.mp4"
depth_5="./data/kinect/pratik/5/depth_raw_cm/video.mp4"

color_6="./data/kinect/pratik/6/color_raw/video.mp4"
depth_6="./data/kinect/pratik/6/depth_raw_cm/video.mp4"

stacked_output="./log/kinect/ref_seq.mp4"
# ffmpeg -i $color_1 -i $depth_1 -i $color_2 -i $depth_2 -i $color_3 -i $depth_3 -i $color_4 -i $depth_4 -i $color_5 -i $depth_5 -i $color_6 -i $depth_6 -filter_complex "
ffmpeg -i $depth_1 -i $depth_2 -i $depth_3 -i $depth_4 -i $depth_5 -i $depth_6 -filter_complex "
[0:v][1:v][2:v][3:v][4:v][5:v]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h1|w0+w1_h2|[out]
" -map "[out]" $stacked_output
