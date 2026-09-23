echo "Running coupled nanopillar examples with varied lamin content"
pitchArray=(0.0 1.5 3.0 4.5 6.0)
radiusArray=(0.0 0.2 0.2 0.2 0.2)
forceArray=(0.0 400.0 800.0)
t0Array=(1000.0 1000.0 1000.0)
laminArray=(1.0 1.0 1.0 1.0 1.0)
FActinArray=(0.0 2.0 6.0 20.0 60.0)
laminPhosArray=(0.0 0.0003 0.003) # default is 0.001
for idx1 in 0 2 4;
do
    for idx2 in 0 1;
    do
        for idx3 in 0 1 2 3 4;
        do
            for idx4 in 0 1 2;
            do
                echo "Running simulation for force=${forceArray[idx2]} and lamin density ${laminArray[idx3]}"
                python3 -u main.py --submit-tscc mechanotransduction_coupled \
                --outdir /root/scratch/results_nanopillarsVaryLaminCond_$(date +%F)/nanopillars_r${radiusArray[idx1]}_p${pitchArray[idx1]}_force${forceArray[idx2]}_t0${t0Array[idx2]}_actin${FActinArray[idx3]}_laminPhos${laminPhosArray[idx4]} \
                --a0-npc 1.0 --force-val ${forceArray[idx2]} --t0-deform ${t0Array[idx2]} \
                --nanopillar-radius ${radiusArray[idx1]} --nanopillar-spacing ${pitchArray[idx1]} --nanopillar-height 1.5 \
                --lamin-diff 0.0 --actin-boost ${FActinArray[idx3]} --lamin-abundance ${laminArray[idx3]} --phiE 0.5 --k-phos ${laminPhosArray[idx4]}
                sleep 1
            done
        done
    done
done

echo "Done."
