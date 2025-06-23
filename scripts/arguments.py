import sys
from pathlib import Path
import argparse


here = Path(__file__).parent.absolute()


def mechanotransduction_example(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "model-files").as_posix())
    import mech_parser_args

    mech_parser_args.add_mechanotransduction_arguments(parser)

def mechanotransduction_example_gq(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "model-files").as_posix())
    import mech_parser_args

    mech_parser_args.add_mechanotransduction_gq_arguments(parser)

def mechanotransduction_nucOnly_example(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "model-files").as_posix())
    import mech_parser_args

    mech_parser_args.add_mechanotransduction_nucOnly_arguments(parser)


def preprocess_mech_mesh(parser: argparse.ArgumentParser):
    sys.path.insert(0, (here / ".." / "model-files").as_posix())
    import mech_parser_args

    mech_parser_args.add_preprocess_mech_mesh_arguments(parser)



def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cg = parser.add_argument_group("cluster", "Options relevant for clusters")
    cluster = cg.add_mutually_exclusive_group()
    cluster.add_argument(
        "--submit-tscc",
        action="store_true",
        help="Add this flag if you want to submit the job on the hopper cluster at TSCC",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Convert notebooks
    subparsers.add_parser("convert-notebooks", help="Convert notebooks to python files")

    # Mechanotransduction example
    preprocess_mech_mesh_parser = subparsers.add_parser(
        "mechanotransduction-preprocess", help="Preprocess mesh for mechanotransduction example"
    )
    preprocess_mech_mesh(preprocess_mech_mesh_parser)

    mechanotransduction_parser = subparsers.add_parser(
        "mechanotransduction", help="Run mechanotransduction example"
    )
    mechanotransduction_example(mechanotransduction_parser)

    mechanotransduction_gq_parser = subparsers.add_parser(
        "mechanotransduction_gq", help="Run mechanotransduction example"
    )
    mechanotransduction_example_gq(mechanotransduction_gq_parser)

    mechanotransduction_nucOnly_parser = subparsers.add_parser(
        "mechanotransduction-nuc-only", help="Run mechanotransduction example with pore formation"
    )
    mechanotransduction_nucOnly_example(mechanotransduction_nucOnly_parser)
    return parser