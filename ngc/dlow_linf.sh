#!/bin/bash

for STEP in 2
do
    for EPS in 0.5
    do
        for EPOCH in 40 50
        do
            #finetune qz
            INS="dgx1v.16g.1.norm"

            BASENAME="step_${STEP}_${EPS}_${EPOCH} ml.model.adv_agentformer"

            WS_ID=yulong-avg # replace with your workspace ID

            NAME="$BASENAME.dlow.finetune.qz"

            CMD="cd /workspace/adv_pred/; git pull; \
                apt-get update && apt-get install libgl1 -y; \
                pip install -r requirements.txt;\
                pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
                export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
            CMD="${CMD} \ 
                python train_dlow.py --cfg adv_mini_nusc_5sample \
                --exp_name linf/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_True_ctx_False_adv \
                --pred_epoch $EPOCH --ngc"

            echo "$CMD"

            ngc batch run \
            --instance "$INS" \
            --name "$NAME" \
            --image "nvidia/pytorch:21.02-py3" \
            --workspace yulong-avg:/workspace \
            --result /result \
            --port 8888 \
            --commandline "$CMD" 

            #finetune
            INS="dgx1v.16g.1.norm"

            WS_ID=yulong-avg # replace with your workspace ID


            NAME="$BASENAME.dlow.finetune"
            CMD="cd /workspace/adv_pred/; git pull; \
                apt-get update && apt-get install libgl1 -y; \
                pip install -r requirements.txt;\
                pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
                export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
            CMD="${CMD} \ 
                python train_dlow.py --cfg adv_mini_nusc_5sample \
                --exp_name linf/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_False_adv \
                --pred_epoch $EPOCH --ngc"

            echo "$CMD"

            ngc batch run \
            --instance "$INS" \
            --name "$NAME" \
            --image "nvidia/pytorch:21.02-py3" \
            --workspace yulong-avg:/workspace \
            --result /result \
            --port 8888 \
            --commandline "$CMD" 

            # ctx finetune
            INS="dgx1v.16g.1.norm"


            WS_ID=yulong-avg # replace with your workspace ID


            NAME="$BASENAME.dlow.finetune"
            CMD="cd /workspace/adv_pred/; git pull; \
                apt-get update && apt-get install libgl1 -y; \
                pip install -r requirements.txt;\
                pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
                export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
            CMD="${CMD} \ 
                python train_dlow.py --cfg adv_mini_nusc_5sample \
                --exp_name linf/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_True_adv \
                --pred_epoch $EPOCH --ngc"

            echo "$CMD"

            ngc batch run \
            --instance "$INS" \
            --name "$NAME" \
            --image "nvidia/pytorch:21.02-py3" \
            --workspace yulong-avg:/workspace \
            --result /result \
            --port 8888 \
            --commandline "$CMD" 

            # qz
            INS="dgx1v.16g.1.norm"


            WS_ID=yulong-avg # replace with your workspace ID

            NAME="$BASENAME.dlow.qz"

            CMD="cd /workspace/adv_pred/; git pull; \
                apt-get update && apt-get install libgl1 -y; \
                pip install -r requirements.txt;\
                pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
                export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
            CMD="${CMD} \ 
                python train_dlow.py --cfg adv_mini_nusc_5sample \
                --exp_name linf/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_True_ctx_False_adv \
                --pred_epoch $EPOCH --ngc"

            echo "$CMD"

            ngc batch run \
            --instance "$INS" \
            --name "$NAME" \
            --image "nvidia/pytorch:21.02-py3" \
            --workspace yulong-avg:/workspace \
            --result /result \
            --port 8888 \
            --commandline "$CMD" 

            # adv train
            INS="dgx1v.16g.1.norm"


            WS_ID=yulong-avg # replace with your workspace ID


            NAME="$BASENAME.dlow"
            CMD="cd /workspace/adv_pred/; git pull; \
                apt-get update && apt-get install libgl1 -y; \
                pip install -r requirements.txt;\
                pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
                export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
            CMD="${CMD} \ 
                python train_dlow.py --cfg adv_mini_nusc_5sample \
                --exp_name linf/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_False_adv \
                --pred_epoch $EPOCH --ngc"

            echo "$CMD"

            ngc batch run \
            --instance "$INS" \
            --name "$NAME" \
            --image "nvidia/pytorch:21.02-py3" \
            --workspace yulong-avg:/workspace \
            --result /result \
            --port 8888 \
            --commandline "$CMD" 

            # ctx
            INS="dgx1v.16g.1.norm"


            WS_ID=yulong-avg # replace with your workspace ID


            NAME="$BASENAME.dlow"
            CMD="cd /workspace/adv_pred/; git pull; \
                apt-get update && apt-get install libgl1 -y; \
                pip install -r requirements.txt;\
                pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
                export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
            CMD="${CMD} \ 
                python train_dlow.py --cfg adv_mini_nusc_5sample \
                --exp_name linf/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_True_adv \
                --pred_epoch $EPOCH --ngc"

            echo "$CMD"

            ngc batch run \
            --instance "$INS" \
            --name "$NAME" \
            --image "nvidia/pytorch:21.02-py3" \
            --workspace yulong-avg:/workspace \
            --result /result \
            --port 8888 \
            --commandline "$CMD" 
        done
    done
done