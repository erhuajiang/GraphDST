#!/bin/bash

# Parameters ------------------------------------------------------

#TASK="sim-m"
#DATA_DIR="data/simulated-dialogue/sim-M"
#TASK="sim-r"
#DATA_DIR="data/simulated-dialogue/sim-R"
#TASK="woz2"
#DATA_DIR="data/woz2"
TASK="multiwoz21"
DATA_DIR="/home/yfeng/graph-DST/dataset/MULTIWOZ2.1"
#DATA_DIR="/home/yfeng/graph-DST/GraphDST/data"

# Project paths etc. ----------------------------------------------

OUT_DIR=/home/yfeng/graph-DST/GraphDST_output/multiwoz2.1/
#OUT_DIR=/home/yfeng/graph-DST/GraphDST_output/test/

# Main ------------------------------------------------------------

for step in dev test; do
    args_add=""
    if [ "$step" = "train" ]; then
	args_add="--do_train --predict_type=dummy"
    elif [ "$step" = "dev" ] || [ "$step" = "test" ]; then
	args_add="--do_eval --predict_type=${step}"
    fi

    CUDA_VISIBLE_DEVICES=0 python3 run_dst.py \
	    --task_name=${TASK} \
	    --data_dir=${DATA_DIR} \
	    --dataset_config=dataset_config/${TASK}.json \
	    --model_type="bert" \
	    --model_name_or_path="bert-base-uncased" \
	    --do_lower_case \
	    --learning_rate=1e-4 \
	    --num_train_epochs=50 \
	    --max_seq_length=180 \
	    --per_gpu_train_batch_size=48 \
	    --per_gpu_eval_batch_size=1 \
	    --output_dir=${OUT_DIR} \
	    --save_epochs=20 \
	    --logging_steps=10 \
	    --warmup_proportion=0.1 \
	    --adam_epsilon=1e-6 \
	    --label_value_repetitions \
            --swap_utterances \
	    --append_history \
	    --use_history_labels \
	    --delexicalize_sys_utts \
	    --class_aux_feats_inform \
	    --class_aux_feats_ds \
	    --seed 42 \
	    --mlm_pre \
	    --mlm_during \
	    ${args_add} \
	    # --no_cuda \
        2>&1 | tee ${OUT_DIR}/${step}.log

    if [ "$step" = "dev" ] || [ "$step" = "test" ]; then
    	python3 metric_bert_dst.py \
    		${TASK} \
		dataset_config/${TASK}.json \
    		"${OUT_DIR}/pred_res.${step}*json" \
    		2>&1 | tee ${OUT_DIR}/eval_pred_${step}.log
    fi
done
