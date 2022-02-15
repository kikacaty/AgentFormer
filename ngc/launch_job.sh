#!/bin/bash
helpFunction()
{
   echo ""
   echo "Usage: $0 -case CASE -name NAME -instance INSTANCE"
   echo -e "\t-c GAN or Pred"
   echo -e "\t-n name of the training session"
   echo -e "\t-i Instance of NGC (16g or 32g)"
   echo -e "\t-t Running time, e.g. 48h"
   exit 1 # Exit script after printing help
}

while getopts "c:n:i:" opt
do
   case "$opt" in
      n ) NAME="$OPTARG" ;;
      a ) ATTACKER="$OPTARG" ;;
      m ) MODE="$OPTARG" ;;
      s ) STEP="$OPTARG" ;;
      i ) INSTANCE="$OPTARG" ;;
      t ) RUNTIME="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

ATTACKER="${ATTACKER:-"adv_noise"}"
MODE="${MODE:-"NORMAL"}"
STEP="${STEP:-"1"}"
INSTANCE="${INSTANCE:-"16g"}"
RUNTIME="${RUNTIME:-"48h"}"



if [ "$CASE" == "Pred" ];then
    NAME="${NAME:-bs_256_adam_ml_model_l5kit}"
else
    NAME="${NAME:-bs_256_adam_ml_model_l5kit_GAN}"
fi


if [ "$INSTANCE" == "32g" ];then
    INS="dgx1v.32g.1.norm"
else
    INS="dgx1v.16g.1.norm"
fi

# if [ -z "$CASE" ] || [ -z "$NAME" ] || [ -z "$INSTANCE" ]
# then
#    echo "Some or all of the parameters are empty";
#    helpFunction
# fi

# Begin script in case all parameters are correct
echo "mode: $MODE"
echo "step: $STEP"
echo "instance: $INS"


WS_ID=CPn3OVBiQ1Gz5U7dDN8B-w # replace with your workspace ID
WS_MOUNT_POINT=/workspace/adv_pred/
DS_MOUNT_POINT=/workspace/adv_pred/datasets/
RESULT_DIR=/workspace/adv_pred/results/
# CMD="cd $WS_MOUNT_POINT; pip install -e .; pip install numpy==1.21.4;\
# python scripts/train_l5kit.py --config_file=/tbsim-ws/tbsim/experiments/templates/l5_raster_plan.json --output_dir $RESULT_DIR --name $NAME --dataset $DS_MOUNT_POINT/lyft_prediction --remove_exp_dir \
# & tensorboard --logdir $RESULT_DIR --bind_all"

if [ "$MODE" == "MIX" ]; 
then 
CMD="cd $WS_MOUNT_POINT; pip install -r requirements.txt;\
pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; \
python adv_train.py --cfg nuscenes_5sample_agentformer_pre --adv_cfg $ATTACKER --pgd_step $STEP --mix"
else 
CMD="cd $WS_MOUNT_POINT; pip install -e .; pip install numpy==1.21.4;\
pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; \
python adv_train.py --cfg nuscenes_5sample_agentformer_pre --adv_cfg $ATTACKER --pgd_step $STEP"
fi 

echo "$CMD"

ngc batch run \
 --instance "$INS" \
 --name "$NAME" \
 --image "nvcr.io/nvidian/nvr-av/tbsim:latest" \
 --workspace "$WS_ID":"$WS_MOUNT_POINT" \
 --result "$RESULT_DIR" \
 --port 8888 \
 --commandline "$CMD"
