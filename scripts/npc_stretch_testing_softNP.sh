echo "Running nanopillar examples testing NPC stretch"
# Create meshes with symm considerations
# indentArray=\
# (0.0 0.2 0.4 0.6 0.8\
#  1.0 1.2 1.4 1.6 1.8\
#  2.0 2.2 2.4 2.6 2.8)
indentArray=\
(0.0 0.4 0.8\
 1.0 1.4 1.8\
 2.0 2.4 2.8)
a0Array=(0 2.5 5.0 7.5 10.0)
for outer_idx in 0 1 2 3 4; # 4 5 6;
do
    for idx in 0 1 2 3 4 5 6 7 8;
    do
        echo "Running simulation for u0 = ${u0Array[outer_idx]} and indentation=${indentArray[idx]}"
        python3 main.py --ntasks 1 --submit-tscc mechanotransduction \
        --mesh-folder /root/shared/gitrepos/smart-mechanotransduction/meshes/nanopillars_indent/nanopillars_indent${indentArray[idx]} \
        --time-step 0.01 --e-val 2000 \
        --outdir /root/scratch/results_nanopillars_indentation_softNP/nanopillars_indent${indentArray[idx]}_a0_${a0Array[outer_idx]}\
        --reaction-rate-on-np 1 --curv-sens 1 --WASP-rate 0.01 \
        --a0-npc ${a0Array[outer_idx]} --nuc-compression ${indentArray[idx]} --endo-rate 10.0
        sleep 100
    done
done

echo "Done."
