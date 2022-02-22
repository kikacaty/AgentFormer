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
while getopts "a:s:" opt
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
NAME="step-${STEP}-${ATTACKER}-${MODE} ml.model.adv_agentformer"
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
echo "name: $NAME"

WS_ID=yulong-avg # replace with your workspace ID
WS_MOUNT_POINT=/workspace/adv_pred/
DS_MOUNT_POINT=/workspace/adv_pred/datasets/
RESULT_DIR=/workspace/adv_pred/results/


if [ "$MODE" == "MIX" ]; 
then 
CMD="cd $WS_MOUNT_POINT; \
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html;\
pip install -r requirements.txt;\
pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; \
python adv_train.py --cfg nuscenes_5sample_agentformer_pre --adv_cfg $ATTACKER --pgd_step $STEP --mix 0.5"
else 
CMD="cd $WS_MOUNT_POINT; \
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html;\
pip install -r requirements.txt;\
pip install wandb; wandb login 66e53af18f876c79bad2f274a73b1c8026ced2ef; \
export WANDB_APIKEY=66e53af18f876c79bad2f274a73b1c8026ced2ef; \
python adv_train.py --cfg nuscenes_5sample_agentformer_pre --adv_cfg $ATTACKER --pgd_step $STEP"
fi 

echo "$CMD"

ngc batch run \
 --instance "$INS" \
 --name "$NAME" \
 --image "nvidia/pytorch:20.02-py3" \
 --workspace yulong-avg:/workspace \
 --result /result \
 --port 8888 \
 --commandline "$CMD"