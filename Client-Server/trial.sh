#!/bin/bash

work_path=/home/endeleze/Desktop/UNav_develop
trials_path=$work_path/Localization/trials.json
plan_path=$work_path/Mapping/data/floor_plan
output=$work_path/Client-Server/trials
script=$work_path/Client-Server/trial.py

python $script --trials_path $trials_path --plan_path $plan_path --output $output

