#!/bin/bash

for STEP in 2
do
    for EPOCH in 40 50
    do
        INS="dgx1v.16g.1.norm"

        BASENAME="eval_${STEP}_${EPOCH} ml.model.adv_agentformer"

        WS_ID=yulong-avg # replace with your workspace ID


        # adv train
        NAME="$BASENAME.base"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_False_adv --pred_epoch ${EPOCH} --adv;\
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_False_adv --pred_epoch ${EPOCH}; "

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
        NAME="$BASENAME.finetune"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_False_adv --pred_epoch ${EPOCH} --adv;\
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_False_adv --pred_epoch ${EPOCH}; "

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD"

        # adv train qz
        NAME="$BASENAME.qz"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_True_ctx_False_adv --pred_epoch ${EPOCH} --adv;\
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_True_ctx_False_adv --pred_epoch ${EPOCH}; "

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD"

        # adv finetune qz
        NAME="$BASENAME.finetune.qz"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_True_ctx_False_adv --pred_epoch ${EPOCH} --adv;\
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_True_ctx_False_adv --pred_epoch ${EPOCH}; "

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD"

        # adv train ctx
        NAME="$BASENAME.ctx"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_True_adv --pred_epoch ${EPOCH} --adv;\
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_True_adv --pred_epoch ${EPOCH}; "

        echo "$CMD"

        ngc batch run \
        --instance "$INS" \
        --name "$NAME" \
        --image "nvidia/pytorch:21.02-py3" \
        --workspace yulong-avg:/workspace \
        --result /result \
        --port 8888 \
        --commandline "$CMD"

        # adv finetune ctx
        NAME="$BASENAME.finetune.ctx"
        CMD="cd /workspace/adv_pred/; git pull; \
            apt-get update && apt-get install libgl1 -y; \
            pip install -r requirements.txt;\
            pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
            export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; "
        CMD="${CMD} \ 
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_True_adv --pred_epoch ${EPOCH} --adv;\
            python test.py --cfg adv_mini_nusc_5sample \
                            --eps 0.5 --pgd_step 20 --ngc\
                            --exp_name linf/all/finetune_0.1/eps_${EPS}_step_${STEP}_free_False_fixed_False_qz_False_ctx_True_adv --pred_epoch ${EPOCH}; "

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
