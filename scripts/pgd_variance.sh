for SEED in 0 1 2 3 4
do
    python test.py --cfg mini_nuscenes_5sample_agentformer_pre \
        --adv --eps 0.1 --pgd_step 10 --sample --seed $SEED \
        $@
done