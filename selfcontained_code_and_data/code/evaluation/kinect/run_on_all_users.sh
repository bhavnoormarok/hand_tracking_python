#!/bin/bash

declare -a users=("aarush" "aniruddha" "avirup" "deepshikha" "parag" "pramit" "pratik" "rahul" "sandika" "shivali" "sukanya" "vihaan")

for user in "${users[@]}"
do
    for sequence in 1 2 3 4 5 6
    do
        if [ "$user" = "aniruddha" ] && [ $sequence -eq 1 ]; then
            continue
        fi
        python code/evaluation/kinect/register_amano.py $user $sequence
        python code/evaluation/kinect/register_mano.py $user $sequence
    done
done