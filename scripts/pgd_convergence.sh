for EPS in 0.2 0.5 1.0
do
    for STEP in 5 10 20 30
    do
        python test.py --cfg mini_nuscenes_5sample_agentformer_pre \
            --adv --eps $EPS --pgd_step $STEP --sample \
            $@
    done
done