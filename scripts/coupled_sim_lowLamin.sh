echo "Running coupled nanopillar examples"
pitchArray=(0.0 2.0 3.0 4.0 5.0 6.0)
radiusArray=(0.0 0.2 0.2 0.2 0.2 0.2)
forceArray=(0.0 400.0 400.0 800.0 800.0)
t0Array=(100.0 100.0 1000.0 100.0 1000.0)
diffArray=(0.001 0.001 0.01 0.01)
actinArray=(0.0 20.0 0.0 20.0)
for idx1 in 0 1 2 3 4 5;
do
    for idx2 in 0 1 2 3 4;
    do
        for idx3 in 0 1 2 3;
        do
            echo "Running simulation for force=${forceArray[idx2]}"
            python3 -u main.py --submit-tscc mechanotransduction_coupled \
            --outdir /root/scratch/results_nanopillars_0phiE_$(date +%F)_70Lamin/nanopillars_r${radiusArray[idx1]}_p${pitchArray[idx1]}_force${forceArray[idx2]}_t0${t0Array[idx2]}_diff${diffArray[idx3]}_actin${actinArray[idx3]} \
            --a0-npc 5.0 --force-val ${forceArray[idx2]} --t0-deform ${t0Array[idx2]} \
            --nanopillar-radius ${radiusArray[idx1]} --nanopillar-spacing ${pitchArray[idx1]} --nanopillar-height 1.5 \
            --lamin-diff ${diffArray[idx3]} --actin-boost ${actinArray[idx3]} --lamin-abundance 0.7 --kcycle 0.0 --phiE 0.5
            sleep 1
        done
    done
done

echo "Done."
