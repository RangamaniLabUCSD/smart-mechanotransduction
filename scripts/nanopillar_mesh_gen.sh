echo "Saving meshes for nanopillar examples"
# Create meshes with symm considerations
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
# cellRadArray=\
# (22.02 22.64 24.33 12.93 15.14 19.69\
#  24.60 14.64 25.64 16.19 26.42)
# cellRadArray=\
# (17.53 18.03 19.37 10.29 12.06 15.68\
#  19.59 11.66 20.42 12.89 21.04)
cellRadArray=\
(20.25 18.52 16.55 19.93 18.04 15.39\
 20.01 17.45 18.06 17.64\
 22.48 18.08 15.39 14.18 12.33)
for idx in 0 1 2 5 6 7 10 11 12 13 14;
do
    # if (${heightArray[idx]}==1.0);
    # then
    echo "Writing mesh for h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]} nanopillars for cellRad=${cellRadArray[idx]}"
    python3 main.py mechanotransduction-preprocess --hEdge 0.5 --hInnerEdge 0.5 \
--mesh-folder /root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars_finalCalcCoarse/nanopillars_h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]}_cellRad${cellRadArray[idx]} \
--contact-rad ${cellRadArray[idx]} \
--nanopillar-radius ${radiusArray[idx]} --nanopillar-height ${heightArray[idx]} --nanopillar-spacing ${pitchArray[idx]}
    # else
    #     echo "Skipping h=${heightArray[idx]}"
    # fi
done
echo "Done."