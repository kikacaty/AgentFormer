#!/bin/bash

for STEP in 2
do
    for EPS in 0.5
    do
        # qz train
        INS="dgx1v.16g.1.norm"

        BASENAME="step_${STEP}_${EPS} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        NAME="$BASENAME.qz.all"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
            --pgd_step $STEP --eps $EPS --test_pgd_step 20 --qz --ngc --all"

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD" 

        # qz finetune
        INS="dgx1v.16g.1.norm"

        BASENAME="step_${STEP}_${EPS} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        NAME="$BASENAME.qz.finetune.all"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
            --pgd_step $STEP --eps $EPS --test_pgd_step 20 --qz --ngc --finetune_lr 0.1 --all\
            --pretrained /workspace/results/adv_mini_nusc_5sample_pre/models/model_0100.p"

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD" 

        # base adv train
        INS="dgx1v.16g.1.norm"

        BASENAME="step_${STEP}_${EPS} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        NAME="$BASENAME.base.all"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
            --pgd_step $STEP --eps $EPS --test_pgd_step 20 --ngc --all"

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD" 

        # adv finetune
        INS="dgx1v.16g.1.norm"

        BASENAME="step_${STEP}_${EPS} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        NAME="$BASENAME.finetune.all"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
            --pgd_step $STEP --eps $EPS --test_pgd_step 20 --ngc --finetune_lr 0.1 --all\
            --pretrained /workspace/results/adv_mini_nusc_5sample_pre/models/model_0100.p "

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD"

        # adv train context
        INS="dgx1v.16g.1.norm"

        BASENAME="step_${STEP}_${EPS} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        NAME="$BASENAME.base.context.all"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
            --pgd_step $STEP --eps $EPS --test_pgd_step 20 --ngc --context --all"

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD" 

        # adv finetune context
        INS="dgx1v.16g.1.norm"

        BASENAME="step_${STEP}_${EPS} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        NAME="$BASENAME.finetune.context.all"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
            --pgd_step $STEP --eps $EPS --test_pgd_step 20 --ngc --finetune_lr 0.1 --all --context\
            --pretrained /workspace/results/adv_mini_nusc_5sample_pre/models/model_0100.p "

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
