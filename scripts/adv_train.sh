python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
    --pgd_step 2 --eps 0.5 --test_pgd_step 20 \
     $@