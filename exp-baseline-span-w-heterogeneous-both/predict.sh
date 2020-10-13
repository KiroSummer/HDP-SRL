#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

MODEL_PATH="./cpb1.0_model"

INPUT_PATH="../data/chn_srl/dev.chinese.cpb1.0.jsonlines"
INPUT_DEP_PATH="/data2/qrxia/dependency-parsing/biaffine-parser/exp-pctb7/CPB1.0_auto_deps/dev.auto.char.dep.conll;/data2/qrxia/dependency-parsing/biaffine-parser/exp-cdt//CPB1.0_auto_deps/dev.auto.char.dep.conll"
GOLD_PATH="../data/chn_srl/dev/dev.props"
OUTPUT_PATH="../temp/cpb.devel.out"

#INPUT_PATH="../data/chn_srl/test.chinese.cpb1.0.jsonlines"
#INPUT_DEP_PATH="/data2/qrxia/dependency-parsing/biaffine-parser/exp-pctb7/CPB1.0_auto_deps/test.auto.char.dep.conll;/data2/qrxia/dependency-parsing/biaffine-parser/exp-cdt//CPB1.0_auto_deps/test.auto.char.dep.conll"
#GOLD_PATH="../data/chn_srl/tst/tst.props"
#OUTPUT_PATH="../temp/cpb.test.out"

CUDA_VISIBLE_DEVICES=$1  ~/anaconda2/bin/python2 ../src/baseline_w_heterogeneous_both/predict.py \
  --span="span" \
  --model="$MODEL_PATH" \
  --input="$INPUT_PATH" \
  --input_dep_trees="$INPUT_DEP_PATH" \
  --output="$OUTPUT_PATH" \
  --gold="$GOLD_PATH" \
  --gpu=$1

