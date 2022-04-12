#!/bin/bash

for STEP in 1 2 3 4 5
do
    for EPS in 0.5 1.0
    do
        INS="dgx1v.16g.1.norm"

        BASENAME="step_${STEP}_${EPS} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        NAME="base-$BASENAME"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
            --pgd_step $STEP --eps $EPS --test_pgd_step 10 --ngc"

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
