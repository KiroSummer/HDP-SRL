export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

CONFIG="config.json"
MODEL="cpb1.0_model"

TRAIN_PATH="../data/chn_srl/train.chinese.cpb1.0.jsonlines"
TRAIN_DEP="/data2/qrxia/dependency-parsing/biaffine-parser/exp-pctb7/CPB1.0_auto_deps/train.5fold.auto.char.dep.conll;/data2/qrxia/dependency-parsing/biaffine-parser/exp-cdt/CPB1.0_auto_deps/train.auto.char.dep.conll"
DEP_TREES="/data2/qrxia/data/chinese/dependency/pctb7-stanford-auto-crf-postag-from-zhli/train.lt100.shuf.crfpos-fltd-0.01-10-fold.conll;/data2/qrxia/data/chinese/dependency/cdt/cdt2-train-ctb-pos.conll"
DEV_PATH="../data/chn_srl/dev.chinese.cpb1.0.jsonlines"
DEV_DEP="/data2/qrxia/dependency-parsing/biaffine-parser/exp-pctb7/CPB1.0_auto_deps/dev.auto.char.dep.conll;/data2/qrxia/dependency-parsing/biaffine-parser/exp-cdt//CPB1.0_auto_deps/dev.auto.char.dep.conll"
GOLD_PATH="../data/chn_srl/dev/dev.props"

gpu_id=$1
CUDA_VISIBLE_DEVICES=$gpu_id ~/anaconda2/bin/python2 ../src/baseline_w_heterogeneous_both/train.py \
   --config=$CONFIG \
   --span="span" \
   --model=$MODEL \
   --train=$TRAIN_PATH \
   --dep_trees=$DEP_TREES \
   --train_dep_trees=$TRAIN_DEP \
   --dev=$DEV_PATH \
   --dev_dep_trees=$DEV_DEP \
   --gold=$GOLD_PATH \
   --gpu=$1
