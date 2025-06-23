import sys
import argparse
from pathlib import Path
import dolfin as d
import numpy as np
from smart import mesh_tools

from mech_parser_args import (
    add_preprocess_mech_mesh_arguments,
)

def main(
    mesh_folder: str,
    hEdge: float = 0.6,
    hInnerEdge: float = 0.6,
    nanopillar_radius: float = 0,
    nanopillar_height: float = 0,
    nanopillar_spacing: float = 0,
    contact_rad: float = 13.0,
    nuc_compression: float = 0.0,
    sym_fraction: float = 1/8,
):
    here = Path(__file__).parent.absolute()
    sys.path.append((here / ".." / "utils").as_posix())

    import spread_cell_mesh_generation as mesh_gen

    hNP = hEdge * 0.3
    nanopillars = [nanopillar_radius, nanopillar_height, nanopillar_spacing]
    cell_mesh, facet_markers, cell_markers, substrate_markers, curv_markers, u_nuc, a_nuc = mesh_gen.create_3dcell(
                                                                        contactRad=contact_rad,
                                                                        hEdge=hEdge, hInnerEdge=hInnerEdge,
                                                                        hNP=hNP,
                                                                        nanopillars=nanopillars,
                                                                        return_curvature=True,
                                                                        sym_fraction=sym_fraction,
                                                                        nuc_compression=nuc_compression)

    mesh_folder = Path(mesh_folder)
    mesh_folder.mkdir(exist_ok=True, parents=True)
    mesh_file = mesh_folder / "spreadCell_mesh.h5"
    mesh_tools.write_mesh(cell_mesh, facet_markers, cell_markers, mesh_file, [substrate_markers])
    d.File(str(mesh_folder / "facets.pvd")) << facet_markers
    d.File(str(mesh_folder / "cells.pvd")) << cell_markers
    # save curvatures for reference
    curv_file_name = mesh_folder / "curvatures.xdmf"
    with d.XDMFFile(str(curv_file_name)) as curv_file:
        curv_file.write(curv_markers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_preprocess_mech_mesh_arguments(parser)
    args = vars(parser.parse_args())
    raise SystemExit(main(**args))