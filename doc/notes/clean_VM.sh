#!/usr/bin/bash

kill -9 $(ps aux | grep torchrun | grep -v grep | awk '{print $2}')
kill -9 $(ps aux | grep wandb | grep -v grep | awk '{print $2}')
kill -9 $(ps aux | grep python3 | grep -v grep | awk '{print $2}')

nvidia-smi
ps uax | grep python3