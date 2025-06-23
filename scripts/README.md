# Scripts

In this folder we provide an example scripts for running sets of simulations on an HPC cluster, or locally. These instructions are the framework for argument parsing are directly adapted from the [smart-comp-sci repository](https://github.com/RangamaniLabUCSD/smart-comp-sci/tree/main/scripts).

usage: main.py [-h] [--submit-tscc]
               {convert-notebooks,mechanotransduction-preprocess,mechanotransduction,mechanotransduction-nuc-only}

    convert-notebooks   Convert notebooks to python files
    mechanotransduction-preprocess
                        Preprocess mesh for mechanotransduction example
    mechanotransduction
                        Run mechanotransduction example with cell on nanopillars
    mechanotransduction-nuc-only
                        Run mechanotransduction example with cell on nanopillars, only considering YAP/TAZ transport in and out of nucleus

Each script has many arguments associated with testing conditions such as nuclear indentation, N-WASP reaction rate, etc. The full list with all default values defined can be found in `mech_parser_args.py` within the `model-files` folder.

There are currently 3 ways to execute the scripts. Examples of such calls can be found in the assorted bash scripts in this folder.

1. By running the script without any additional flags, e.g
    ```
    python3 main.py mechanotransduction [args]
    ```
    will run the script directly as a normal script. `[args]` is a series of arguments giving the specifications for a given simulation. For instance, to specify a nuclear indentation of 2.8, `--nuc-compression 2.8` would be appended.
2. You can submit a job to an HPC cluster by adjusting the SLURM script in `runner.py` and passing the `--submit-tscc` (or another custom) flag, e.g
    ```
    python3 main.py --submit-tscc mechanotransduction [args]
    ```
    Rather than running the script directly, this will generate a SLURM job script (see `runner.py`) for submission to an HPC cluster.
3. You can navigate to the example folders and run the notebooks directly using `jupyter`


## Setup environment 
All the code in this repository depends on [`smart`](https://rangamanilabucsd.github.io/smart), which in turn depends on the [development version of legacy FEniCs](https://bitbucket.org/fenics-project/dolfin/src/master/). While `smart` is a pure python package and can be easily installed with `pip` (i.e `python3 -m pip install fenics-smart`), the development version of FEniCs can be tricky to install, and we recommend to use [docker](https://www.docker.com) for running the code locally, or [singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) to run on a cluster. Alternatively, you can setup an environment for running on HPC clusters using Spack as described in the [smart-comp-sci repository](https://github.com/RangamaniLabUCSD/smart-comp-sci/tree/main/scripts).

### Set up environment using docker

We provide a pre-built docker image containing both the development version of FEniCS and smart which you can pull using 
```
docker pull ghcr.io/rangamanilabucsd/smart:2.2.3
```
If you prefer to run the code in jupyter notebooks we also provide a docker image for that
```
docker pull ghcr.io/rangamanilabucsd/smart-lab:2.2.3
```
You can read more about how to initialize a container and running the code in the [smart documentation](https://rangamanilabucsd.github.io/smart/README.html#installation).


## Running the scripts

All the scripts are available as Jupyter notbooks. If you want to run the examples using the `main.py` script in the following folder, you need to convert the notebooks to python files first. 


### Convert notebooks to python files
In order to run the scripts on the cluster we need to first convert the notebooks to python files. To do this we will use `jupytext` which is part of the requirements. To convert all the notebooks into python files you can do
```
python3 main.py convert-notebooks
```
inside this folder (called `scripts`)