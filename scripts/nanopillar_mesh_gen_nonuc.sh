echo "Saving meshes for nanopillar examples"
# Create meshes with symm considerations
radiusArray=\
(0.1 0.1 0.1\
 0.5 0.25\
 0.0 0.0 0.0 0.0 0.0\
 0.5)
pitchArray=\
(5.0 2.5 1.0\
 5.0 2.5\
 0.0 0.0 0.0 0.0 0.0\
 3.5)
heightArray=\
(1.0 1.0 1.0\
 1.0 1.0\
 0.0 0.0 0.0 0.0 0.0\
 3.0)
cellRadArray=\
(20.25 18.52 16.55\
 20.01 17.45\
 22.48 18.08 15.39 14.18 12.33\
 15.5)
for idx in 0 1 2 3 4 5 6 7 8 9 10;
do
    echo "Writing mesh for h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]} nanopillars for cellRad=${cellRadArray[idx]}"
    python3 main.py mechanotransduction-preprocess --hEdge 0.5 --hInnerEdge 0.5 --sym-fraction 0.25 \
--mesh-folder /root/shared/gitrepos/smart-mechanotransduction/meshes/nanopillars_nonuc/nanopillars_h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]}_cellRad${cellRadArray[idx]} \
--contact-rad ${cellRadArray[idx]} --no-nuc \
--nanopillar-radius ${radiusArray[idx]} --nanopillar-height ${heightArray[idx]} --nanopillar-spacing ${pitchArray[idx]}
done
echo "Done."