echo "Running coupled nanopillar examples"
radiusArray=\
(0.1 0.1 0.1\
 0.5 0.25\
 0.0 0.0 0.0 0.0 0.0)
pitchArray=\
(5.0 2.5 1.0\
 5.0 2.5\
 0.0 0.0 0.0 0.0 0.0)
heightArray=\
(1.0 1.0 1.0\
 1.0 1.0\
 0.0 0.0 0.0 0.0 0.0)
cellRadArray=\
(20.25 18.52 16.55\
 20.01 17.45\
 22.48 18.08 15.39 14.18 12.33)
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
        for outeridx in 0 1 2 3 4 5;
        do
            echo "Running simulation for force=${forceArray[idx1]}"
            python3 -u main.py --submit-tscc mechanotransduction_coupled \
            --mesh-folder /root/shared/gitrepos/smart-mechanotransduction/meshes/nanopillars_nonuc/nanopillars_h${heightArray[outeridx]}_p${pitchArray[outeridx]}_r${radiusArray[outeridx]}_cellRad${cellRadArray[outeridx]} \
            --outdir /root/scratch/results_nanopillars_sweep_$(date +%F)/nanopillars_force${forceArray[idx1]}_a0${a0Array[idx2]}_t0${t0Array[idx2]}_mesh${outeridx} \
            --WASP-rate 0.01 --a0-npc ${a0Array[idx2]} --force-val ${forceArray[idx1]} --t0-deform ${t0Array[idx2]}
            sleep 100
        done
    done
done

for idx1 in 1 2 3;
do
    for idx2 in 0 1 2 3;
    do
        for outeridx in 0 1 2 3 4 5;
        do
            echo "Running simulation for force=${forceArray[idx1]}"
            python3 -u main.py --submit-tscc mechanotransduction_coupled \
            --mesh-folder /root/shared/gitrepos/smart-mechanotransduction/meshes/nanopillars_nonuc/nanopillars_h${heightArray[outeridx]}_p${pitchArray[outeridx]}_r${radiusArray[outeridx]}_cellRad${cellRadArray[outeridx]} \
            --outdir /root/scratch/results_nanopillars_sweep_$(date +%F)/nanopillars_force${forceArray[idx1]}_a0${a0Array[idx2]}_t0${t0Array[idx2]}_mesh${outeridx} \
            --WASP-rate 0.01 --a0-npc ${a0Array[idx2]} --force-val ${forceArray[idx1]} --t0-deform ${t0Array[idx2]}
            sleep 100
        done
    done
done

echo "Done."
