import argparse
from pathlib import Path

def add_mechanotransduction_arguments(parser: argparse.ArgumentParser) -> None:
    # List of mechanotransduction arguments:
    #  (mesh-folder and outdir are provided as strings and converted to pathlib.Path when called as script)
    #   - mesh-folder: pathlib path to current mesh folder 
    #   - outdir: pathlib path to output folder for current simulation
    #   - time-step: starting time step in s (float)
    #   - e-val: substrate stiffness in kPa (float)
    #   - curv-sens: H0 in curvature-sensitive FAK phosphorylation equation (float)
    #   - nuc-compression: nuclear indentation at central nanopillar in microns (float)
    #   - npc-slope: (not used in simulations for paper) - gradient in NPCs from bottom to top of NE 
    #                (0 - no gradient, 1 - ranges from max at bottom to 0 at top), total NPC number is conserved.  (float)
    #   - a0-npc: stretch sensitivity parameter for NPCs (float)
    #   - WASP-rate: rate of N-WASP mediated curvature-sensitive actin assembly at substrate (float)
    #   - endo-rate: baseline rate of integrin endocytosis (float)
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=Path.cwd().parent / "meshes" / "nanopillars_baseline" / "nanopillars_h1.0_p2.5_r0.25_cellRad17.45"
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path("results_mechanotransduction")
    )
    parser.add_argument(
        "-dt",
        "--time-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--e-val",
        type=float,
        default=1e7,
    ) # stiffness for glass coverslip
    parser.add_argument(
        "--curv-sens",
        type=float,
        default=0.0,
    ) # curvature sensitivity factor of FAK phosph.
    parser.add_argument(
        "--nuc-compression",
        type=float,
        default=0.0,
    ) # nuc indentation on nanopillars
    parser.add_argument(
        "--npc-slope",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--a0-npc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--WASP-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--endo-rate",
        type=float,
        default=1.0,
    )

def add_mechanotransduction_gq_arguments(parser: argparse.ArgumentParser) -> None:
    # List of mechanotransduction_gq arguments, same as normal mech, but with addition args:
    #   - mutant: whether or not current case is a Gq mutant (Bool)
    #   - TRIO-ref: sets RhoA activation sensitivity to TRIO
    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=Path.cwd().parent / "meshes" / "nanopillars_baseline" / "nanopillars_h1.0_p2.5_r0.25_cellRad17.45"
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path("results_mechanotransduction")
    )
    parser.add_argument(
        "-dt",
        "--time-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--e-val",
        type=float,
        default=1e7,
    ) # stiffness for glass coverslip
    parser.add_argument(
        "--curv-sens",
        type=float,
        default=0.0,
    ) # curvature sensitivity factor of FAK phosph.
    parser.add_argument(
        "--nuc-compression",
        type=float,
        default=0.0,
    ) # nuc indentation on nanopillars
    parser.add_argument(
        "--npc-slope",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--a0-npc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--WASP-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--endo-rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--mutant",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--TRIO-ref",
        type=float,
        default=0.01,
    )


def add_nucmech_arguments(parser: argparse.ArgumentParser) -> None:
    # List of nuclear mechanics arguments:
    #  (mesh-folder and outdir are provided as strings and converted to pathlib.Path when called as script)
    #   - mesh-folder: pathlib path to current mesh folder 
    #   - outdir: pathlib path to output folder for current simulation
    #   - max-force: value of overall force to which to run this test (float)
    #   - start-force: value of force to start simulation at (float)
    #   - u0: starting displacement (from previous iteration perhaps) (dolfin function or xdmf file name)
    #   - bulk-mod: effective bulk modulus to enforce incompressibility (float)
    #   - nanopillar-radius: rNP in microns (float)
    #   - nanopillar-height: hNP in microns (float)
    #   - nanopillar-spacing: pNP (pitch) in microns (float)

    parser.add_argument(
        "--mesh-folder",
        type=Path,
        default=Path.cwd().parent / "meshes" / "nanopillars_baseline" / "nanopillars_h1.0_p2.5_r0.25_cellRad17.45"
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path("results_nucmech")
    )
    parser.add_argument(
        "-max-force",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--start-force",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--u0",
        type=Path,
        default=Path.cwd().parent / "u0.xdmf",
    )
    parser.add_argument(
        "--bulk-mod",
        type=float,
        default=1e6,
    )
    parser.add_argument("--nanopillar-radius", type=float, default=0.25)
    parser.add_argument("--nanopillar-height", type=float, default=1.0)
    parser.add_argument("--nanopillar-spacing", type=float, default=2.5)


def add_preprocess_mech_mesh_arguments(parser: argparse.ArgumentParser) -> None:
    # List of mechanotransduction_nucOnly arguments:
    #   (mesh-folder, outdir, and full-sims-folder are provided as strings 
    #    and converted to pathlib.Path when called as script)
    #   - mesh-folder: pathlib path to current mesh folder
    #   - hEdge: mesh resolution at PM (in microns) (float)
    #   - hInnerEdge: mesh resolution at NE (in microns) (float)
    #   - nuc-compression: nuclear indentation at central nanopillar in microns (float)
    #   - nanopillar-radius: rNP in microns (float)
    #   - nanopillar-height: hNP in microns (float)
    #   - nanopillar-spacing: pNP (pitch) in microns (float)
    #   - contact-rad: cell contact radius in microns (float)
    #   - sym-fraction: fraction of cell to simulate accouting for symmetry (float)
    parser.add_argument("--mesh-folder", type=Path,
        default=Path.cwd().parent / "meshes" / "nanopillars_baseline" / "nanopillars_h1.0_p2.5_r0.25_cellRad17.45")
    parser.add_argument("--hEdge", type=float, default=0.6)
    parser.add_argument("--hInnerEdge", type=float, default=0.6)
    parser.add_argument("--nanopillar-radius", type=float, default=0.25)
    parser.add_argument("--nanopillar-height", type=float, default=1.0)
    parser.add_argument("--nanopillar-spacing", type=float, default=2.5)
    parser.add_argument("--contact-rad", type=float, default=17.45)
    parser.add_argument("--nuc-compression", type=float, default=0.0)
    parser.add_argument("--sym-fraction", type=float, default=1/8)