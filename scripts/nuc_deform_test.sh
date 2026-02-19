echo "Running coupled nanopillar examples"
# radiusArray=\
# (0.1 0.1 0.1\
#  0.5 0.25\
#  0.0 0.0 0.0 0.0 0.0)
# pitchArray=\
# (5.0 2.5 1.0\
#  5.0 2.5\
#  0.0 0.0 0.0 0.0 0.0)
# heightArray=\
# (1.0 1.0 1.0\
#  1.0 1.0\
#  0.0 0.0 0.0 0.0 0.0)
# cellRadArray=\
# (20.25 18.52 16.55\
#  20.01 17.45\
#  22.48 18.08 15.39 14.18 12.33)
# --mesh-folder /root/shared/gitrepos/smart-mechanotransduction/meshes/nanopillars_nonuc/nanopillars_h${heightArray[idx1]}_p${pitchArray[idx2]}_r${radiusArray[idx3]} \
heightArray=(1.5 3.0)
pitchArray=(1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0)
radiusArray=(0.2 0.5)

# first run flat case
python3 -u main.py --submit-tscc nuc_mechanics \
--outdir /root/scratch/results_nanopillars_nuconly_$(date +%F)_sweep/nucmech_flat \
--max-force 1000.0  --nanopillar-radius 0.0 --nanopillar-height 0.0 --nanopillar-spacing 0.0 --nuc-only
sleep 1
for idx1 in 0 1;
do
    for idx2 in 0 1 2 3 4 5 6 7 8 9 10 11;
    do
        for idx3 in 0 1;
        do
            python3 -u main.py --submit-tscc nuc_mechanics \
            --outdir /root/scratch/results_nanopillars_nuconly_$(date +%F)_sweep/nucmech_nanopillars_h${heightArray[idx1]}_p${pitchArray[idx2]}_r${radiusArray[idx3]} \
            --max-force 1000.0  --nanopillar-radius ${radiusArray[idx3]} --nanopillar-height ${heightArray[idx1]} --nanopillar-spacing ${pitchArray[idx2]} --nuc-only
            sleep 1
        done
    done
done

echo "Done."
