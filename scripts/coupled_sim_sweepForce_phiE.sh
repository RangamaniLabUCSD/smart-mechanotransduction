echo "Running coupled nanopillar examples"
pitchArray=(0.0 3.0 6.0 0.0 3.0 6.0 0.0 3.0 6.0 0.0 3.0 6.0)
radiusArray=(0.0 0.2 0.2 0.0 0.2 0.2 0.0 0.2 0.2 0.0 0.2 0.2)
a0Array=(0.0 0.0 0.0 0.5 0.5 0.5 1.0 1.0 1.0 2.0 2.0 2.0)
forceArray=(0.0 200.0 400.0 600.0 800.0 1000.0)
for idx1 in 0 1 2 3 4 5 6 7 8 9 10 11;
do
    for idx2 in 0 1 2 3 4 5;
    do
        echo "Running simulation for force=${forceArray[idx2]}"
        python3 -u main.py --submit-tscc mechanotransduction_coupled \
        --outdir /root/scratch/results_nanopillars_sweepForce_$(date +%F)_revision_a0_${a0Array[idx1]}/nanopillars_r${radiusArray[idx1]}_p${pitchArray[idx1]}_force${forceArray[idx2]} \
        --a0-npc ${a0Array[idx1]} --force-val ${forceArray[idx2]} --t0-deform 1000.0 \
        --nanopillar-radius ${radiusArray[idx1]} --nanopillar-spacing ${pitchArray[idx1]} --nanopillar-height 1.5 \
        --lamin-diff 0.0 --actin-boost 30.0 --lamin-abundance 1.0 --phiE 0.5
        sleep 1
    done
done

echo "Done."
