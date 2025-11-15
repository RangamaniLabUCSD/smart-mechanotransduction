echo "Running coupled nanopillar examples"
forceArray=\
(0.0 500.0 700.0 900.0)
a0Array=\
(2.0 2.0 5.0 5.0)
t0Array=\
(100.0 1000.0 100.0 1000.0)
for idx1 in 0;
do
    for idx2 in 0;
    do
        echo "Running simulation for force=${forceArray[idx1]}"
        python3 -u main.py --submit-tscc mechanotransduction_coupled \
        --mesh-folder /root/shared/gitrepos/smart-mechanotransduction/meshes/nanopillars_nonuc/nanopillars_h3.0_p3.5_r0.5_cellRad15.5 \
        --outdir /root/scratch/results_nanopillars_largeNP_$(date +%F)/nanopillars_force${forceArray[idx1]}_a0${a0Array[idx2]}_t0${t0Array[idx2]} \
        --WASP-rate 0.01 --a0-npc ${a0Array[idx2]} --force-val ${forceArray[idx1]} --t0-deform ${t0Array[idx2]}
        sleep 100
    done
done

for idx1 in 1 2 3;
do
    for idx2 in 0 1 2 3;
    do
        echo "Running simulation for force=${forceArray[idx1]}"
        python3 -u main.py --submit-tscc mechanotransduction_coupled \
        --mesh-folder /root/shared/gitrepos/smart-mechanotransduction/meshes/nanopillars_nonuc/nanopillars_h3.0_p3.5_r0.5_cellRad15.5 \
        --outdir /root/scratch/results_nanopillars_largeNP_$(date +%F)/nanopillars_force${forceArray[idx1]}_a0${a0Array[idx2]}_t0${t0Array[idx2]} \
        --WASP-rate 0.01 --a0-npc ${a0Array[idx2]} --force-val ${forceArray[idx1]} --t0-deform ${t0Array[idx2]}
        sleep 100
    done
done

echo "Done."
