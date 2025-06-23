# smart-mechanotransduction

This code provides a general framework for simulations of mechanotransduction using [SMART (Spatial Modeling Algorithms for Reaction and Transport)](https://github.com/RangamaniLabUCSD/smart.git).
See more info about running the code and reproducing the results in the [scripts](scripts) folder.

## Installation

Running this code requires SMART, version 2.2.3 or later.
To run the scripts, we advice usage of docker, and the following base image
`ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30`, which after installation of docker, can be started with

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared -p 8888:8888 --name smart-comp-sci  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
```

This should preferably be started from the root of this git repo, as `-v` shared the current directory on your computer with the docker container.

This will launch a terminal with [FEniCS](https://bitbucket.org/fenics-project/dolfin/src/master/) installed.
To install the compatible version of SMART, call

```bash
python3 -m pip install fenics-smart[lab]==2.2.3 -U
```
Alternatively, you can use the provided docker image from smart directly, i.e
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared -p 8888:8888 --name smart-comp-sci  ghcr.io/rangamanilabucsd/smart-lab:v2.2.3
```

To run notebooks in your browser, call

```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

## Repository contents

This repository is organized into several subfolders, `model-files`, `scripts`, and `utils`. The files contained in each are briefly summarized below.

Notably, prior to running any of the examples, meshes must be generated locally or files must be downloaded from this [this repository](https://doi.org/10.5281/zenodo.13948827). Each individual folder of npy files and `simulation_results_2.8indent`should be place in a folder `analysis_data` and all meshes (in separate folders `nanopillars_baseline`, `nanopillars_indent`, and `nanopillars_movenuc`) should be placed in a folder `meshes`.

### model-files
- `mechanotransduction.ipynb`: Main model file providing the specifications for YAP/TAZ signaling in cells on nanopillar substrates, either with or without nuclear deformation.
- `mechanotransduction_withgq.ipynb`: Model file that also includes Gq signaling pathway.
- `mech_parser_args.py`: Contains names and default values for all input arguments needed to run each script. See this file for definitions of all arguments.
- `pre_process_mesh.py`: Script called to generate meshes.
We also include mechanical models for nuclear deformations:
- `nuclear_deformation_axisymm_mixed.py`: Solve axisymmetric nuclear deformation on a nanopillar using mixed pressure-displacement formulation
- `nuclear_deformation_mixed3d.py`: Solve nuclear deformation on nanopillar array using mixed pressure-displacement formulation 

### mesh-files
- `interp_meshes.ipynb`: Notebook used to interpolate from partial mesh (e.g., 1/8th) to full cell mesh and/or to interpolate over different time points from those simulated.
- `spread_cell_mesh_generation.py`: Functions used for mesh generation

### plotting
- `mech_figs.ipynb`: Used to generate plots for figures
- `smart_analysis.py`: functions used for postprocessing of XDMF files after running simulation (load in vectors and compute spatial averages)
- `smart_plots.mplstyle`: specifications for matplotlib

### scripts
Note that this folder contains its own README to describe the workflow for running simulations in this repository. Most of the infrastructure used here is inherited from the [`smart-comp-sci` repository](https://github.com/RangamaniLabUCSD/smart-comp-sci).
- `arguments.py`: script for adding arguments using argparse
- `main.py`: python file used to run scripts (see README in scripts folder)
- `runner.py`: all functions used when calling different scripts and/or generating SLURM file
The folder also considers a series of bash scripts used to generate meshes and run simulations shown in the main paper.
- `nanopillar_mesh_gen.sh`: generate meshes for cells on nanopillar substrates
- `nanopillar_mesh_gen_deform.sh`: generate meshes with deformed nuclei on nanopillars
- `run_mechanotransduction_nanopillar.sh`: run baseline mechanotransduction simulations for nanopillar geometries (on cluster)
- `run_mechanotransduction_nanopillar_soft.sh`: run baseline mechanotransduction simulations for nanopillar geometries, soft material (on cluster)
- `run_gq_mechanotransduction.sh`: run baseline mechanotransduction simulations including Gq signaling
- `npc_stretch_testing.sh`: run simulations for deformed nuclear geometries
- `npc_stretch_testing_softNP.sh`: run simulations for deformed nuclear geometries on soft nanopillars

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg