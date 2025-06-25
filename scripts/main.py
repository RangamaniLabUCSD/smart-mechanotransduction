from typing import NamedTuple, Callable
import arguments
import runner

class Command(NamedTuple):
    msg: str
    script: Callable


commands = {
    "convert-notebooks": Command(
        msg="Convert notebook",
        script=runner.convert_notebooks
    ),
    "mechanotransduction-preprocess": Command(
        msg="Run preprocess mechanotransduction mesh",
        script=runner.preprocess_mech_mesh
    ),
    "mechanotransduction": Command(
        msg="Run mechanotransduction example",
        script=runner.mechanotransduction_example
    ),
    "mechanotransduction_gq": Command(
        msg="Run mechanotransduction example",
        script=runner.mechanotransduction_example_gq
    ),
    "nuc_mechanics": Command(
        msg="Run nuclear mechanics simulation",
        script=runner.nuc_mechanics
    ),
}

def main():
    parser = arguments.setup_parser()
    args = vars(parser.parse_args())

    try:
        cmd = commands[args["command"]]
        print(cmd.msg)
        cmd.script(**args)
    except KeyError:
        raise ValueError(f"Invalid command: {args['command']}")


if __name__ == "__main__":
    raise SystemExit(main())
