echo "Running nanopillar examples"
radiusArray=\
(0.1 0.1 0.1 0.1 0.1 0.1\
 0.5 0.25 0.5 0.25\
 0.0 0.0 0.0 0.0 0.0)
pitchArray=\
(5.0 2.5 1.0 5.0 2.5 1.0\
 5.0 2.5 5.0 2.5\
 0.0 0.0 0.0 0.0 0.0)
heightArray=\
(1.0 1.0 1.0 3.0 3.0 3.0\
 1.0 1.0 3.0 3.0\
 0.0 0.0 0.0 0.0 0.0)
cellRadArray=\
(20.25 18.52 16.55 19.93 18.04 15.39\
 20.01 17.45 18.06 17.64\
 22.48 18.08 15.39 14.18 12.33)
EModArray=\
(2000 2000 2000 2000 2000\
 2000 2000 2000 2000 2000)
curv0Array=(0 2)
endoRateArray=(10.0 10.0)
for outeridx in 0 1;
do
    for idx in 0 1 2 5 6 7;
    do
        echo "Running simulation for h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]} nanopillars for cellRad=${cellRadArray[idx]} and EMod=${EModArray[idx]}"
        python3 main.py --submit-tscc mechanotransduction \
        --mesh-folder /root/shared/gitrepos/smart-mechanotransduction/meshes/nanopillars_baseline/nanopillars_h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]}_cellRad${cellRadArray[idx]} \
        --time-step 0.01 --e-val ${EModArray[idx]} \
        --outdir /root/scratch/results_nanopillars_withIntegrinSigm_zeroRecycle_soft/nanopillarsWithIntegrin_h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]}_cellRad${cellRadArray[idx]}_curvSens${curv0Array[outeridx]}_endo${endoRateArray[outeridx]} \
        --reaction-rate-on-np 1.0 --curv-sens ${curv0Array[outeridx]} --WASP-rate 0.01 --endo-rate ${endoRateArray[outeridx]}
        sleep 30
    done
done

echo "Done."
