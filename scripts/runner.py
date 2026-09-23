import sys
import json
from textwrap import dedent
from pathlib import Path
import subprocess as sp

tscc_template = dedent(
    """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --partition=condo
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt
#SBATCH --account=csd786
#SBATCH --qos=condo
#SBATCH --mem=10G

module load singularitypro/3.11
module load mpich/ge/gcc/64/3.4.2
cd /tscc/nfs/home/eafrancis/gitrepos/smart-mechanotransduction/scripts

SCRATCH_DIRECTORY=/tscc/lustre/ddn/scratch/eafrancis/nanopillar-sims
mkdir -p ${{SCRATCH_DIRECTORY}}
echo "Scratch directory: ${{SCRATCH_DIRECTORY}}"

echo 'Run command in container: python {script} {args}'
singularity exec --bind $HOME:/root/shared,\
$TMPDIR:/root/tmp,$SCRATCH_DIRECTORY:/root/scratch \
/tscc/nfs/home/eafrancis/smart-newmeshview.sif \
python3 {script} {args}

# Move log file to results folder
mv ${{SLURM_JOBID}}-* ${{SCRATCH_DIRECTORY}}
"""
)

here = Path(__file__).parent.absolute()


def run(
    args,
    script: str,
    submit_tscc: bool,
    dry_run: bool = False,
    job_name: str = "",
):
    in_args = list(map(str, args))
    args_str = " ".join(in_args)
    if dry_run:
        print(f"Run command: {sys.executable} {script} {args_str}")
        return

    if submit_tscc:
        template = tscc_template
    else:
        sp.run([sys.executable, script, *in_args])
        return

    job_file = Path("tmp_job.sbatch")
    job_file.write_text(
        template.format(
            job_name=job_name,
            script=script,
            args=args_str,
        )
    )
    sp.run(["sbatch", job_file.as_posix()])
    job_file.unlink()

def nuc_mechanics(
    mesh_folder: Path,
    outdir: Path,
    max_force: float = 0.01,
    start_force: float = 0.0,
    u0: Path = Path(""),
    dry_run: bool = False,
    submit_tscc: bool = False,
    bulk_mod: float = 1e6,
    nanopillar_radius: float = 0.25,
    nanopillar_height: float = 1.0,
    nanopillar_spacing: float = 2.5,
    contactRad: float = 17.45,
    nuc_only: bool = False,
    **kwargs,
):
    args = [
        "--mesh-folder",
        Path(mesh_folder).as_posix(),
        "--max-force",
        max_force,
        "--start-force",
        start_force,
        "--u0",
        u0,
        "--bulk-mod",
        bulk_mod,
        "--nanopillar-radius",
        nanopillar_radius,
        "--nanopillar-height",
        nanopillar_height,
        "--nanopillar-spacing",
        nanopillar_spacing,
        "--contactRad",
        contactRad,
    ]

    if nuc_only:
        args.append("--nuc-only")

    args.extend(["--outdir", Path(outdir).as_posix()])

    script = (
        (here / ".." / "model-files" / "nuc_mech_only.py")
        .absolute()
        .resolve()
        .as_posix()
    )
    run(
        job_name="nucmech",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_tscc=submit_tscc,
    )

def coupled_example(
    mesh_folder: Path,
    outdir: Path,
    time_step: float,
    e_val: float,
    dry_run: bool = False,
    submit_tscc: bool = False,
    curv_sens: float = 2.0,
    npc_slope: float = 0.0,
    a0_npc: float = 0.0,
    WASP_rate: float = 0.0,
    endo_rate: float = 1.0,
    force_val: float = 0.0,
    t0_deform: float = 100.0,
    actin_boost: float = 20.0,
    actin_decay: float = 0.2,
    lamin_diff: float = 0.001,
    lamin_abundance: float = 1.0,
    k_phos: float = 0.001,
    phiE: float = 0.0,
    nanopillar_radius: float = 0.25,
    nanopillar_spacing: float = 2.5,
    nanopillar_height: float = 1.0,
    **kwargs,
):
    args = [
        "--mesh-folder",
        Path(mesh_folder).as_posix(),
        "--time-step",
        time_step,
        "--e-val",
        e_val,
        "--curv-sens",
        curv_sens,
        "--npc-slope",
        npc_slope,
        "--a0-npc",
        a0_npc,
        "--WASP-rate",
        WASP_rate,
        "--endo-rate",
        endo_rate,
        "--force-val",
        force_val,
        "--t0-deform",
        t0_deform,
        "--actin-boost",
        actin_boost,
        "--actin-decay",
        actin_decay,
        "--lamin-diff",
        lamin_diff,
        "--lamin-abundance",
        lamin_abundance,
        "--k-phos",
        k_phos,
        "--phiE",
        phiE,
        "--nanopillar-radius",
        nanopillar_radius,
        "--nanopillar-height",
        nanopillar_height,
        "--nanopillar-spacing",
        nanopillar_spacing,
    ]

    args.extend(["--outdir", Path(outdir).as_posix()])

    script = (
        (here / ".." / "model-files" / "mechanotransduction_nucmech.py")
        .resolve()
        .as_posix()
    )
    run(
        job_name="mechanotransduction",
        args=args,
        dry_run=dry_run,
        script=script,
        submit_tscc=submit_tscc,
    )


def convert_notebooks(dry_run: bool = False, **kwargs):
    import jupytext

    for example_folder in (here / "..").iterdir():
        # Only look in folders that contains the word model
        if "model" not in example_folder.stem:
            continue

        # Loop over all files in that folder
        for f in example_folder.iterdir():
            if not f.suffix == ".ipynb":
                continue

            print(f"Convert {f} to {f.with_suffix('.py')}")
            if dry_run:
                continue

            text = jupytext.reads(f.read_text())
            jupytext.write(text, f.with_suffix(".py"))
