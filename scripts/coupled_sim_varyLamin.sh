echo "Running coupled nanopillar examples with varied lamin content"
pitchArray=(0.0 3.0 6.0)
radiusArray=(0.0 0.2 0.2)
forceArray=(0.0 400.0 400.0 800.0 800.0)
t0Array=(100.0 100.0 1000.0 100.0 1000.0)
laminArray=(0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
for idx1 in 0 1 2;
do
    for idx2 in 0 1 2;
    do
        for idx3 in 0 1 2 3 4 5 6 7 8 9 10;
        do
            echo "Running simulation for force=${forceArray[idx2]} and lamin density ${laminArray[idx3]}"
            python3 -u main.py --submit-tscc mechanotransduction_coupled \
            --outdir /root/scratch/results_nanopillarsVaryLamin_$(date +%F)/nanopillars_r${radiusArray[idx1]}_p${pitchArray[idx1]}_force${forceArray[idx2]}_t0${t0Array[idx2]}_lamin${laminArray[idx3]} \
            --a0-npc 5.0 --force-val ${forceArray[idx2]} --t0-deform ${t0Array[idx2]} \
            --nanopillar-radius ${radiusArray[idx1]} --nanopillar-spacing ${pitchArray[idx1]} --nanopillar-height 1.5 \
            --lamin-diff 0.001 --actin-boost 20.0 --lamin-abundance ${laminArray[idx3]} --phiE 0.5
            sleep 1
        done
    done
done

echo "Done."
