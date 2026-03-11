[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18690946.svg)](https://doi.org/10.5281/zenodo.18690946)

# smart-mechanotransduction

This code provides a general framework for simulations of mechanotransduction using [SMART (Spatial Modeling Algorithms for Reaction and Transport)](https://github.com/RangamaniLabUCSD/smart.git).
See more info about running the code and reproducing the results in the [scripts](scripts) folder.

## Installation

Running this code requires the v2.3.1.alpha.2 pre-release of SMART available [here](https://github.com/RangamaniLabUCSD/smart/tree/v2.3.1.alpha.2).
The simplest way to use this version of SMART is to use the provided docker image. 
You can get this image by pulling it from the github registry
```
docker pull ghcr.io/rangamanilabucsd/smart:v2.3.1.alpha.2
```

In order to start a container you can use the [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command. For example the command
```
docker run --rm -v $(pwd):/home/shared -w /home/shared -ti ghcr.io/rangamanilabucsd/smart:v2.3.1.alpha.2
```
will run the latest version and share your current working directory with the container.
The source code of smart is located at `/repo` in the docker container.
More options for installing SMART are detailed in [the SMART repository](https://github.com/RangamaniLabUCSD/smart.git).

## Repository contents

This repository is organized into several subfolders, `model-files`, `mesh-files`, `scripts`, and `plotting`. The files contained in each are briefly summarized below.

### model-files
The relevant files for nuclear mechanics simulations include:
- `nuclear_deformation_mixed3d.py`: Solve nuclear deformation on nanopillar array using mixed pressure-displacement formulation
- `mechanotransduction_nucmech.ipynb`: Main model file providing the specifications for mechanotransduction signaling coupled to nuclear mechanics simulations.
- `mech_parser_args.py`: Contains names and default values for all input arguments needed to run each script. See this file for definitions of all arguments.

### mesh-files
- `spread_cell_mesh_generation.py`: Functions used for mesh generation

### plotting
- `mech_figs.ipynb`: Used to generate plots for figures of coupled model
- `nuc_deform_plots.ipynb`: Used to generate plots for figures of nuclear mechanics model
- `smart_analysis.py`: functions used for postprocessing of XDMF files after running simulation (load in vectors and compute spatial averages)
- `smart_plots.mplstyle`: specifications for matplotlib
- `koushki_data.txt`: Data from [Koushki et al 2023 PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC10334804/), Fig 3C 

### scripts
Note that this folder contains its own README to describe the workflow for running simulations in this repository. Most of the infrastructure used here is inherited from the [`smart-comp-sci` repository](https://github.com/RangamaniLabUCSD/smart-comp-sci).
- `arguments.py`: script for adding arguments using argparse
- `main.py`: python file used to run scripts (see README in scripts folder)
- `runner.py`: all functions used when calling different scripts and/or generating SLURM file
The folder also considers a series of bash scripts used to generate meshes and run simulations shown in the main paper.

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
