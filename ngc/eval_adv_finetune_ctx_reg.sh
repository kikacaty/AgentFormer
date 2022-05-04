#!/bin/bash

for STEP in 2
do
    for BETA in 0.01 0.1 0.5 1.0 2.0 10.0
    do
        EPS=0.5
        INS="dgx1v.16g.1.norm"

        BASENAME="eval_${STEP}_${LR} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID

        NAME="$BASENAME.ctx.reg.finetune"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        for EPOCH in 10 20 30 40 50
        do
            CMD="${CMD} \ 
                python test.py --cfg adv_mini_nusc_5sample \
                                --eps 0.5 --pgd_step 20 --ngc\
                                --exp_name finetune_0.1/ctx_reg_${BETA}/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_False_adv --pred_epoch ${EPOCH} --adv;\
                python test.py --cfg adv_mini_nusc_5sample \
                                --eps 0.5 --pgd_step 20 --ngc\
                                --exp_name finetune_0.1/ctx_reg_${BETA}/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_False_adv --pred_epoch ${EPOCH}; "
        done
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
