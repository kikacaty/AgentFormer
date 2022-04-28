python test.py --cfg adv_mini_nusc_5sample \
               --eps 0.5 --pgd_step 20 --adv_DLOW --adv\
               --exp_name fast/finetune_0.1/qz_reg/eps_0.5_step_2_free_False_amp_False_fixed_False_qz_False_adv --pred_epoch 40;\
python test.py --cfg adv_mini_nusc_5sample \
               --eps 0.5 --pgd_step 20 --adv_DLOW\
               --exp_name fast/finetune_0.1/eps_0.5_step_2_free_False_amp_False_fixed_False_qz_False_adv --pred_epoch 50;