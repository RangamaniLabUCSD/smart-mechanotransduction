from mech_parser_args import add_nucmech_arguments
import nuclear_deformation_mixed3d as nucdef
import argparse
import sys
parser = argparse.ArgumentParser()
if len(sys.argv) == 1:
    args = []
else:
    add_nucmech_arguments(parser)
    args = vars(parser.parse_args())
    args["softFactor"] = 1.0
nuc_dict = nucdef.start_nuc_mech(args)
# if not nuc_dict["nuc_only"]:
max_force = 1000.0 #nuc_dict["max_force"]
kInc = 5.0
while nuc_dict["kRamp"][-1] < max_force:
    nuc_dict["kRamp"][-1] = nuc_dict["kRamp"][-2] + kInc
    if nuc_dict["kRamp"][-1] > max_force:
        nuc_dict["kRamp"][-1] = max_force
    print(f"Current force is {nuc_dict['kRamp'][-1]}")
    nuc_dict = nucdef.solve_next_nuc_step(nuc_dict)