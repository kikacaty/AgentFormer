for SEED in 1 2 3 4 5
do
    for EPS in 0.1 0.5 1.0
    do
        for STEP in 1 2 3 4 5 10 20 30
        do
            python test.py --cfg mini_nuscenes_5sample_agentformer \
                --adv --eps $EPS --pgd_step $STEP --seed $SEED \
                $@
        done
    done
done