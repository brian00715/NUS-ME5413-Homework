#!/bin/bash

source ./.venv/bin/activate
python ./src/task1.py --seq $1
mpv ./temp/seq_$1.avi
