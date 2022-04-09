python adv_train.py --cfg adv_mini_nusc_5sample_pre --adv_cfg at_noise \
    --pgd_step 1 --eps 0.1 --test_pgd_step 20 \
     $@