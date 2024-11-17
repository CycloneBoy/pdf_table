#!/bin/bash

set -v
set -e


begin_time=$(date "+%Y_%m_%d_%H_%M_%S")
echo "pdftable train lore begin time = ${begin_time}"

##########################################################################
## 激活环境
## LoreAndLineCell  LineCell Lore LineCellPdf
##########################################################################
nohup python ./tests/trainer/run_table_trainer.py > ~/.pdftable/logs/run_train.log 2>&1 &