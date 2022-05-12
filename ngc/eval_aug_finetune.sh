#!/bin/bash

for EPS in 0.05 0.1 0.2 0.3 0.5 1.0
do
    for EPOCH in 40 50
    do
        INS="dgx1v.16g.1.norm"

        BASENAME="eps_${EPS}_epoch_${EPOCH} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        # adv train
        NAME="$BASENAME.aug"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name finetune_0.1/aug_noise/eps_${EPS} --pred_epoch ${EPOCH} --adv;\
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name finetune_0.1/aug_noise/eps_${EPS} --pred_epoch ${EPOCH}; "

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
