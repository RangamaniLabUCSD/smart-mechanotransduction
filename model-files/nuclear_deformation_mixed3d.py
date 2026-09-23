from dolfin import *
from smart import mesh_tools
import numpy as np
import pathlib
import sys
import argparse, logging
import petsc4py.PETSc as PETSc
from scipy.optimize import lsq_linear, root_scalar

from ufl_legacy.algorithms.ad import expand_derivatives

import petsc4py
petsc4py.init(sys.argv)

from mech_parser_args import add_nucmech_arguments
here = pathlib.Path(__file__).parent
sys.path.insert(0, (here / ".." / "scripts").as_posix())

smart_logger = logging.getLogger("smart")
logger = logging.getLogger("mechanotransduction")
logger.info("Starting nuclear mechanics calculation")

# here = pathlib.Path.cwd() 
sys.path.insert(0, (here / ".." / "mesh-files").as_posix())
import spread_cell_mesh_generation as mesh_gen

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

def check_intersection(fmixed, mesh_ne_outer, mesh_cyto, zShift):
    ucur = fmixed.sub(0)
    Vlin = VectorFunctionSpace(mesh_ne_outer, "P", 1)
    ulin = interpolate(ucur, Vlin)
    ne_outer_coords = Vlin.tabulate_dof_coordinates()
    xDef = np.zeros_like(mesh_ne_outer.coordinates())
    for i in range(len(xDef)):
        vec_idx = [3*i,3*i+1,3*i+2]
        xDef[i] = ne_outer_coords[3*i] + ulin.vector()[vec_idx]
    xDef[:,2] += zShift
    cyto_coord = mesh_cyto.coordinates()
    if np.any(np.max(xDef,0) > np.max(cyto_coord,0)) or np.min(xDef[:,2]) < np.min(cyto_coord[:,2]):
        return True
    # otherwise we need to check one by one
    for i in range(len(xDef)):
        logger.debug(f"Checking for intersections at idx {i}")
        x = xDef[i]
        # move slightly negative points to slightly positive
        if np.abs(x[0]) < 1e-6:
            x[0] = 1e-6
        if np.abs(x[1]) < 1e-6:
            x[1] = 1e-6
        if np.arctan2(x[1],x[0]) > 0.2499*np.pi:
            continue
        inMesh = False
        for c in cells(mesh_cyto):
            if c.contains(Point(x)):
                inMesh = True
                break
        if not inMesh:
            return True
    # If we made it this far, there are no intersections
    return False

# map subfunction to parent mesh
def sub_to_parent(func, parent_mesh):
    if len(func.ufl_shape) > 0:
        vec_logic = True
        num_comp = func.ufl_shape[0]
    else:
        vec_logic = False
        num_comp = 1
    if vec_logic:
        Vnew = VectorFunctionSpace(parent_mesh, "P", 1)
    else:
        Vnew = FunctionSpace(parent_mesh, "P", 1)
    new_func = Function(Vnew)
    Vold = func.function_space()
    boundDict = Vold.mesh().topology().mapping()
    if parent_mesh.id() not in boundDict.keys():
        raise ValueError("Function is not in subdomain of this parent mesh")
    else:
        boundMap = boundDict[parent_mesh.id()].vertex_map()
    old_vals = func.vector().get_local()
    new_vals = new_func.vector().get_local()
    for j in range(len(boundMap)):
        old_sub_idx = vertex_to_dof_map(Vold)[num_comp*j:num_comp*(j+1)]
        new_idx = vertex_to_dof_map(Vnew)[num_comp*boundMap[j]:num_comp*(boundMap[j]+1)]
        new_vals[new_idx] = old_vals[old_sub_idx]
    new_func.vector().set_local(new_vals)
    new_func.vector().apply("insert")
    return new_func

def sub_to_sub(func, new_submesh, parent_mesh):
    Vnew = VectorFunctionSpace(new_submesh, "P", 1)
    new_func = Function(Vnew)
    Vold = func.function_space()
    boundDictOld = Vold.mesh().topology().mapping()
    boundDictNew = Vnew.mesh().topology().mapping()
    if parent_mesh.id() not in boundDictOld.keys():
        raise ValueError("Function is not in subdomain of this parent mesh")
    elif parent_mesh.id() not in boundDictNew.keys():
        raise ValueError("Submesh does not belong to this parent mesh")
    else:
        boundMapOld = boundDictOld[parent_mesh.id()].vertex_map()
        boundMapNew = boundDictNew[parent_mesh.id()].vertex_map()
    old_vals = func.vector().get_local()
    new_vals = new_func.vector().get_local()
    for j in range(len(boundMapOld)):
        old_sub_idx = vertex_to_dof_map(Vold)[3*j:3*(j+1)]
        parent_idx = boundMapOld[j]
        new_mesh_idx = np.nonzero(np.array(boundMapNew) == parent_idx)[0]
        if len(new_mesh_idx) > 1:
            raise ValueError("Could not find unique mapping")
        elif len(new_mesh_idx) == 0:
            continue
        else:
            new_mesh_idx = new_mesh_idx[0]
        new_idx = vertex_to_dof_map(Vnew)[3*new_mesh_idx:3*(new_mesh_idx+1)]
        new_vals[new_idx] = old_vals[old_sub_idx]
    new_func.vector().set_local(new_vals)
    new_func.vector().apply("insert")
    return new_func

def start_nuc_mech(args):

    # parser = argparse.ArgumentParser()
    # add_nucmech_arguments(parser)
    # args = vars(parser.parse_args())
    # run_local = True
    if not isinstance(args, dict):
        args = dict()
        args["mesh_folder"] = pathlib.Path("/root/shared/gitrepos/smart-mechanotransduction"
                                           "/meshes/nanopillars_nonuc/nanopillars_h3.0_p3.5_r0.5_cellRad15.5")
        # args["mesh_folder"] = pathlib.Path("/root/shared/gitrepos/smart-mechanotransduction"
        #                                    "/meshes/nanopillars_nonuc/nanopillars_h1.0_p2.5_r0.25_cellRad17.45")
        args["max_force"] = 0.0
        args["start_force"] = 0.0
        args["u0"] = pathlib.Path("")
        args["bulk_mod"] = 1e8
        args["nanopillar_radius"] = 0.2
        args["nanopillar_height"] = 3.0
        args["nanopillar_spacing"] = 6.0 #7.0 #3.5
        args["contactRad"] = 15.5
        args["outdir"] = pathlib.Path(f"/root/scratch/nuc_test_h3")#tallerPartialSlip")
        args["nuc_only"] = True
        args["softFactor"] = 1.0
        args["bulk_mod"] *= args["softFactor"]
    
    nuc_only = args["nuc_only"]
    testPressBoost = False
    z0 = 1.04

    # Create mesh and define function space
    # soft_factor = 5
    nucRad1 = 4.1#5.09#6.625#5.5
    nucRad2 = 4.1#5.09#3.0
    thickness = 0.2
    rthickness = thickness
    zthickness = thickness
    NE_layers = 1
    if args["nanopillar_radius"] == 0:
        hEdge = 0.15
    else:
        hEdge = args["nanopillar_radius"]/1.5
        if hEdge > 0.15:
            hEdge = 0.15
    zRoof = np.inf
    zRoofAlt = 10.0

    mesh_ref, mf2, mf3 = mesh_gen.NE_mesh(rRad=nucRad1, zRad=nucRad2, thickness=[rthickness,zthickness],
                                        hEdge=hEdge, hInnerEdge=hEdge, sym_fraction=0.25, NE_layers=NE_layers,
                                        use_tmp=True, hTop=2*hEdge)
    mesh_ref.coordinates()[:,2] += z0
    mesh = create_meshview(mf3, 1)
    inner_mesh = create_meshview(mf3, 2)

    # create 1/8 nucl geometry for mapping to cell geometry
    mesh_ref_eighth, mf2_eighth, mf3_eighth = mesh_gen.NE_mesh(rRad=nucRad1, zRad=nucRad2, thickness=[rthickness,zthickness],
                                        hEdge=hEdge, hInnerEdge=hEdge, sym_fraction=0.25, NE_layers=NE_layers,
                                        use_tmp=True)
    mesh_ref_eighth.coordinates()[:,2] += z0
    ne_mesh_eighth = create_meshview(mf2_eighth, 10)

    # define nanopillar dimensions for current case
    npSpacing = args["nanopillar_spacing"]
    npRad = args["nanopillar_radius"] + 0.05
    if args["nanopillar_radius"] == 0.0:
        hNP = 0.0
    else:
        hNP = args["nanopillar_height"]
    if npSpacing > 0.0:
        xMax = np.ceil(2*nucRad1 / npSpacing) * npSpacing
        xNP = np.arange(0.0, xMax+1e-12, npSpacing)
        yNP = np.arange(0.0, xMax+1e-12, npSpacing)
        xNP, yNP = np.meshgrid(xNP, yNP)
        xNP = xNP.flatten()
        yNP = yNP.flatten()
    else:
        xNP = np.array([])
        yNP = np.array([])

    nanopillar_specs = dict()
    nanopillar_specs["hNP"] = hNP
    nanopillar_specs["rNP"] = npRad
    nanopillar_specs["pNP"] = npSpacing
    nanopillar_specs["xNP"] = xNP
    nanopillar_specs["yNP"] = yNP
    nanopillar_specs["contactRad"] = args["contactRad"]

    mf_surf = MeshFunction("size_t", mesh, 2, 0)
    mf_surfInt = MeshFunction("size_t", mesh, 2, 0)
    mf_dirichlet = MeshFunction("size_t", mesh, 2, 0)
    mf_dirichlet2 = MeshFunction("size_t", mesh, 2, 0)

    # redetermine surface markers
    rad_eff1 = (nucRad1**2 * nucRad2)**(1/3)
    rad_eff2 = ((nucRad1-rthickness)**2 * (nucRad2-zthickness))**(1/3)
    class OuterSurf(SubDomain):
        def inside(self, x, on_boundary):
            bound_val = pow(pow(x[0]/nucRad1,2) + pow(x[1]/nucRad1,2) + 
                        pow((x[2]-nucRad2-z0)/nucRad2,2),0.5)
            cutoffFrac = 0.99#1-0.95*thickness_thresh/rad_eff1
            return bound_val > cutoffFrac and on_boundary
    class InnerSurf(SubDomain):
        def inside(self, x, on_boundary):
            bound_val = pow(pow(x[0]/(nucRad1-rthickness),2) + pow(x[1]/(nucRad1-rthickness),2) +
                        pow((x[2]-nucRad2-z0)/(nucRad2-zthickness),2),0.5)
            cutoffFrac1 = 1.01#1+0.95*thickness_thresh/rad_eff2
            cutoffFrac2 = 0.99#1-0.95*thickness_thresh/rad_eff2
            return  bound_val < cutoffFrac1 and bound_val > cutoffFrac2 and on_boundary

    outerSurf = OuterSurf()
    innerSurf = InnerSurf()
    outerSurf.mark(mf_surf, 10) # mark outer surface points as 10
    innerSurf.mark(mf_surf, 12) # mark inner surface points as 12

    sym_fraction = 0.25
    for f in facets(mesh):
        if mf_surf[f] == 10 or mf_surf[f] == 12:
            if sym_fraction == 1/2:
                if f.midpoint().x() < 1e-6:
                    mf_surf[f] = 0
            elif sym_fraction == 1/4:
                if f.midpoint().x() < 1e-6 or f.midpoint().y() < 1e-6:
                    mf_surf[f] = 0
            else:
                thetaCur = np.arctan2(f.midpoint().y(), f.midpoint().x())
                if f.midpoint().x() < 1e-6 or thetaCur > 0.999*2*np.pi*sym_fraction:
                    mf_surf[f] = 0
    
    mesh_ne_outer = create_meshview(mf_surf, 10)

    class NanopillarContact(SubDomain): # no initial contact
        def inside(self, x, on_boundary):
            return False
    class RoofInt(SubDomain): # for integrating over top
        def inside(self, x, on_boundary):
            bound_val = pow(pow(x[0]/nucRad1,2) + pow(x[1]/nucRad1,2) + 
                        pow((x[2]-nucRad2-z0)/nucRad2,2),0.5)
            cutoffFrac = 0.99#1-0.95*thickness_thresh/rad_eff1
            return (bound_val > cutoffFrac and on_boundary and 
                    (np.sqrt(x[0]**2 + x[1]**2) < 10*hEdge and np.sqrt(x[0]**2 + x[1]**2) > 0.0*nucRad1)
                      and x[2] > (z0 + nucRad2))
    class SymmAxis1(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] < 0.001 and on_boundary
    class SymmAxis2(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < 0.001 and on_boundary
    class SymmAxis3(SubDomain):
        def inside(self, x, on_boundary):
            return np.abs(x[2] - (nucRad2+z0)) < 2*hEdge

    nanopillar = NanopillarContact()
    symm1 = SymmAxis1()
    symm2 = SymmAxis2()
    symm3 = SymmAxis3()
    roofInt = RoofInt()
    nanopillar.mark(mf_dirichlet, 1) # mark nanopillar contact with 1
    array_dirichlet = mf_dirichlet.array()[:]
    array_ref = mf_surf.array()[:]
    array_dirichlet[np.logical_and(array_dirichlet == 1, array_ref != 10)] = 0

    symm1.mark(mf_dirichlet, 2)
    symm2.mark(mf_dirichlet, 3)
    symm3.mark(mf_dirichlet2, 8)
    roofInt.mark(mf_surfInt, 1)
    ds_roofInt = Measure("ds", domain=mesh, subdomain_data=mf_surfInt)
    mf_vol = MeshFunction("size_t", mesh, 3, 1)

    V_vector = FunctionSpace(mesh, VectorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
    # V_tensor = FunctionSpace(mesh, TensorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
    V_scalar = FunctionSpace(mesh, "P", 1)
    normal_expr = Expression(("x[0]/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))",
                              "x[1]/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))",
                              "(x[2]-z0)/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))"),
                            z0=nucRad2+z0, degree=1)
    theta_expr = Expression(("-x[1]/sqrt(pow(x[0],2) + pow(x[1],2) + 0.0001)",
                            "x[0]/sqrt(pow(x[0],2) + pow(x[1],2) + 0.0001)",
                            "0"), degree=1)
    normals = project(normal_expr, V_vector)
    thetas = project(theta_expr, V_vector)
    theta_coords = V_vector.tabulate_dof_coordinates()
    theta_vec = thetas.vector()[:]
    for i in range(0,len(theta_vec),3):
        if theta_coords[i][0] < 1e-6 and theta_coords[i][1] < 1e-6:
            theta_vec[i:i+3] = [0.,1.,0.]
    thetas.vector().set_local(theta_vec)
    thetas.vector().apply("insert") 
    phis = project(cross(normals,thetas), V_vector)
    inner_normals = normals # the same

    x = SpatialCoordinate(mesh)
    results_folder = args["outdir"]
    File(f"{results_folder}/test_shell.pvd") << mesh

    # Define mixed function space for displacements over each region
    el1 = VectorElement("P", mesh.ufl_cell(), 2) # u function space
    el2 = FiniteElement("P", mesh.ufl_cell(), 1) # p function space
    # el3 = FiniteElement("P", mesh_ne_outer.ufl_cell(), 1)
    mixed_element = MixedElement([el1, el2]) # mixed function space
    Vmixed = FunctionSpace(mesh, mixed_element)
    V1 = Vmixed.sub(0)
    V_psi = FunctionSpace(mesh_ne_outer, "P", 1) # psi_c function space for contact mechanics

    # Define Dirichlet boundary conditions
    V = FunctionSpace(mesh, "P", 1)
    zDisplOuterExpr = Expression("-z0-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, r1=nucRad1, r2=nucRad2, z0=z0)
    zDisplOuter = interpolate(zDisplOuterExpr, V)
    zDisplFloorExpr = Expression("-z0-hNP-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, hNP=hNP, r1=nucRad1, r2=nucRad2, z0=z0)
    zDisplFloor = interpolate(zDisplFloorExpr, V)
    uFixedExpr = Expression("0.0", degree=1)
    uxFixed = interpolate(uFixedExpr, V)
    uyFixed = interpolate(uFixedExpr, V)
    bc_nanopillar_z = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
    bc_nanopillar_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 1)
    bc_nanopillar_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 1)
    bc_side_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 8)
    bc_side_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 8)
    bc_floor = DirichletBC(V1.sub(2), zDisplFloor, mf_dirichlet, 7)
    bc_symm1_y = DirichletBC(V1.sub(1), Constant(0.0), mf_dirichlet, 2)
    bc_symm2_x = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 3)
    bc_symm3_z = DirichletBC(V1.sub(2), Constant(0.0), mf_dirichlet2, 8)
    stick = False
    if stick:
        bcs = [bc_nanopillar_z, bc_nanopillar_x, bc_nanopillar_y, 
            bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
    else:
        bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z]

    # Define functions
    (v, q)  = TestFunctions(Vmixed)             # Test function
    fmixed  = Function(Vmixed)                 # Displacement from previous iteration
    (u, p) = split(fmixed)
    psi_c = Function(V_psi)

    domain_id = MeshFunction("size_t", mesh, 2, 0)
    for f in facets(mesh):
        if mf_dirichlet[f] == 2:
            domain_id[f] = 2
        elif mf_dirichlet[f] == 3:
            domain_id[f] = 3
        elif mf_dirichlet[f] == 1:
            domain_id[f] = 11
        elif mf_surf[f] == 10:
            domain_id[f] = 1
        elif mf_surf[f] == 12:
            domain_id[f] = 4
    ds = Measure('ds', domain=mesh, subdomain_data=domain_id)

    # Kinematics
    d = 3#u.geometric_dimension()
    I = variable(Identity(d))             # Identity tensor
    F = variable(I + grad(u))             # Deformation gradient
    C = variable(F.T*F)                   # Right Cauchy-Green tensor
    (u_prev, p_prev) = fmixed.split()
    psi_c_prev = psi_c.copy()

    # Invariants of deformation tensors
    I1 = variable(tr(C))
    I2 = variable(0.5*(I1**2 - tr(C*C)))
    J  = variable(det(F))

    E1, E2 = Constant(5000.0), Constant(1000.0)#/soft_factor)
    E1scale = Function(V_scalar)
    E2scale = Function(V_scalar)
    for Escale in [E1scale,E2scale]:
        Escale.vector()[:] = args["softFactor"]
        Escale.vector().apply("insert")

    # Stored strain energy density (incompressible Mooney Rivlin model)
    psi = E1scale*E1*(I1-3) + E2scale*E2*(I2-3) - p*(J-1)

    zNP = [-0.05]#[-nucRad - 0.05]
    idx = 0
    zMove = 0.1
    zStep = 0.1
    zFinal = zNP[-1] + zMove
    kMin = args["start_force"]
    kMax = args["max_force"]
    kInc = 5.0# / soft_factor #max([0.001, (args["max_force"]-args["start_force"])/1000])
    kRamp = [0.0]#[kMin]
    pMin = 0.0
    pMax = 820.0 * args["softFactor"]# / soft_factor
    baselineOsmotic = 2400.0
    innerBulkMod = 100.0 * args["softFactor"]
    pRamp = [pMin]
    p0 = Constant(0.0)
    dpress = 200.0 * args["softFactor"] # / soft_factor

    u_file = XDMFFile(f"{results_folder}/u_np_ellipsoid.xdmf")
    u_file.parameters["flush_output"] = True
    u_file.write(mesh)
    u_file.write_checkpoint(fmixed.split()[0], "u_np_ellipsoid", idx, append=True)
    ulin_file = XDMFFile(f"{results_folder}/ulin_np_ellipsoid.xdmf")
    ulin_file.parameters["flush_output"] = True
    ulin = project(fmixed.sub(0), V_vector)
    ulin_file.write(ulin, idx)
    
    a_vector = J * dot(normals, inv(F))
    a_vector = project(a_vector, V_vector)
    a_file = XDMFFile(f"{results_folder}/a_np_ellipsoid.xdmf")
    a_file.parameters["flush_output"] = True
    a_file.write(a_vector, idx)

    # save stress and strain energy density for current configuration
    surf_stress_file = XDMFFile(f"{results_folder}/surf_stress.xdmf")
    surf_stress_file.parameters["flush_output"] = True
    psi_file = XDMFFile(f"{results_folder}/psi_fcn.xdmf")
    psi_file.parameters["flush_output"] = True
    vonmises_stress_file = XDMFFile(f"{results_folder}/vonmises_stress.xdmf")
    vonmises_stress_file.parameters["flush_output"] = True
    surf_tension_file = XDMFFile(f"{results_folder}/surf_tension.xdmf")
    surf_tension_file.parameters["flush_output"] = True

    a_scalar = ufl.sqrt(a_vector[0]**2 + a_vector[1]**2 + a_vector[2]**2)
    ds_integrate = Measure('ds', domain=mesh, subdomain_data=mf_surf)
    dx_integrate = Measure('dx', domain=mesh)
    inner_SA = []
    outer_SA = []
    vol = []
    inner_SA_ref = assemble(1.0*ds_integrate(12))
    outer_SA_ref = assemble(1.0*ds_integrate(10))
    vol_ref = assemble(1.0*dx_integrate)

    pumpedUp = False
    inCell = False

    curForce = 0.0
    curPress = pRamp[-1]
    logger.info(f"Starting force is {curForce}")
    logger.info(f"Starting pressure is {curPress}")
    forceConst = Constant(curForce)
    pressConst = Constant(curPress)
    Tdir = Expression(("0.0", "0.0", f"-exp((x[2]-{2*nucRad2}-{z0})/{0.5*nucRad2})"), degree=1)
    n_g = Expression(("0.0", "0.0", "-1.0"), degree=1)
    outer_area_factor = J * dot(normals, inv(F))
    inner_area_factor = J * dot(inner_normals, inv(F))
    theta_area_factor = J * dot(phis, inv(F)) 
    phi_area_factor = J * dot(thetas, inv(F))
    T = forceConst*Tdir*sqrt(inner(outer_area_factor,outer_area_factor))
    Press_in = (p0 + pressConst)*inner_area_factor
    Press_out = p0 * outer_area_factor

    # add steric forces at bottom
    V_vector_P2 = VectorFunctionSpace(mesh, "P", 2)
    Tsteric = Function(V_vector_P2)
    Tsteric_vec = Tsteric.vector()[:]
    stericCoords = V_vector_P2.tabulate_dof_coordinates()
    dSteric = 0.2
    stericMag = Constant(1000.0)
    stericCutoff = 0.01
    nanopillar_logic = Constant(0.0)
    for i in range(len(xNP)):
        nanopillar_logic += exp(-((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)**2/(npRad**4))
    npContactForce = nanopillar_logic*(stericMag*ufl.exp(-(x[2]+u[2])/stericCutoff)*
                      sqrt(inner(outer_area_factor,outer_area_factor))*(-n_g))

    # Convert potential energy to first Piola-Kirchoff stress tensor
    dx = Measure("dx", domain=mesh, subdomain_data=mf_vol)
    Ttensor = diff(psi, F)
    Fvar = (inner(grad(v), Ttensor)*dx(1) - inner(v, T)*ds(1) - inner(v,-Press_out)*ds(1) - inner(v, npContactForce)*ds(1) -
            inner(v, Press_in)*ds(4) - inner(q, args["bulk_mod"]*(J-1) + p) * dx(1))

    surf_stress = project(dot(normals, Ttensor)/sqrt(inner(outer_area_factor,outer_area_factor)), V_vector)
    surf_stress_file.write(surf_stress, idx)
    TCtensor = dot(Ttensor, F.T) / J # cauchy stress tensor
    von_mises_calc = sqrt((TCtensor[0,0] - TCtensor[1,1])**2 + (TCtensor[1,1] - TCtensor[2,2])**2 + (TCtensor[2,2] - TCtensor[0,0])**2
                    + 6*(TCtensor[0,1]**2 + TCtensor[0,2]**2 + TCtensor[2,1]**2))/np.sqrt(2)
    von_mises = project(von_mises_calc, V_scalar)
    vonmises_stress_file.write(von_mises, idx)
    psi_fcn = project(psi, V_scalar)
    psi_file.write(psi_fcn, idx)
    
    surf_tension = project(dot(dot(thetas, Ttensor), thetas)/sqrt(inner(theta_area_factor,theta_area_factor)) + 
                           dot(dot(phis, Ttensor), phis)/sqrt(inner(phi_area_factor,phi_area_factor)), V_scalar)
    surf_tension_file.write(surf_tension, idx)
    # Compute Jacobian of F
    Jvar = derivative(Fvar, fmixed)
    custom_solver = False
    if custom_solver:
        problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
    else:
        solver = init_solver(Fvar, fmixed, bcs, Jvar)

    if nuc_only:
        zRoofPM = 0.0
    else:
        raise ValueError("Only nuc only supported here")
    dSteric = 0.2
    zShift = hNP + dSteric

    vol0_nuc = 142.0 + vol_ref
    vol_tot = vol0_nuc/0.3627 #1024.0
    vol0_cyto = vol_tot - vol0_nuc
    vol_excl_nuc = (vol0_nuc-vol_ref)*0.01
    vol_excl_cyto = vol0_cyto*0.02
    if testPressBoost:
        pressBoost = 0.5*baselineOsmotic
    else:
        pressBoost = 0.0
    fracOsmotic = (pMax-pressBoost)/pMax
    nucSoluteFactor = (fracOsmotic*pMax + baselineOsmotic) * (vol0_nuc - vol_ref - vol_excl_nuc) 
    nucBoostFactor = (pressBoost) * (vol0_nuc - vol_ref)**2
    cytoSoluteFactor = (baselineOsmotic) * (vol0_cyto - vol_excl_cyto)
    def computeBothPress(nucVol):
        cytoVol = vol_tot - nucVol - vol_ref
        pNuc = nucSoluteFactor/(nucVol-vol_excl_nuc) + nucBoostFactor/(nucVol**2)
        pCyto = cytoSoluteFactor/(cytoVol - vol_excl_cyto)
        return pNuc, pCyto
    def computePress(nucVol):
        cytoVol = vol_tot - nucVol - vol_ref
        pTest = (nucSoluteFactor/(nucVol-vol_excl_nuc) + nucBoostFactor/(nucVol**2) - 
                 cytoSoluteFactor/(cytoVol - vol_excl_cyto))
        return pTest
    def rootPress(nucVol, pressVal):
        return computePress(nucVol) - pressVal
    def findVolFromPress(pressVal, x0):
        bracketVals=[0.5*x0, 1.5*x0]
        pressAns = root_scalar(rootPress, args=(pressVal,), bracket=bracketVals)
        return pressAns.root

    V1lin = VectorFunctionSpace(mesh, "P", 1)
    linFcn = interpolate(fmixed.sub(0), V1lin)
    uLagrange, volLagrange = lagrangeMap(mf3, 2, mf2, [10,12], 
                                        [linFcn,linFcn], 1/4, None)
    vol_inner = []

    fmixed_prev = fmixed.vector()[:].copy()
    uEval = fmixed.sub(0)
    uEval.set_allow_extrapolation(True)
    zTops = [zShift + 2*nucRad2 + uEval(0,0,2*nucRad2)[2]]
    reinit = False
    stopLogic = False

    while True: # loop is broken manually below
        keepSwimming = True
        it = 0
        pressIts = [pRamp[-1]]
        idx += 1
        while keepSwimming:
            # need to keep running the solver until the BCs do not change for the current force
            it += 1
            set_step = False
            # try solving, if diverges, take smaller step
            while not set_step:
                if len(kRamp) > 1:
                    curPress = pRamp[-1]
                if not pumpedUp:
                    kRamp[-1] = 0.0
                    curForce = 0.0
                else:
                    curForce = kRamp[-1]
                logger.info(f"Current force is {curForce}")
                logger.info(f"Current pressure is {curPress}")
                forceConst.assign(curForce)
                if pumpedUp and pRamp[-1]!=pMax:
                    nucPress, cytoPress = computeBothPress(volEstIts[-1])
                    pressConst.assign(nucPress-cytoPress)
                    p0.assign(cytoPress)
                else:
                    pressConst.assign(curPress)
                    p0.assign(baselineOsmotic*curPress/pMax)
                reset = False
                try:
                    if custom_solver:
                        solver.solve(None, problem.fmixed.vector().vec())
                        if solver.getConvergedReason() <= 0:
                            reset = True
                        else:
                            set_step = True
                    else:
                        solver.solve()
                        set_step = True
                except:
                    reset = True
                if reset:
                    if pumpedUp:
                        logger.warning(f"Resetting kRamp from {kRamp[-1]} to")
                        if len(kRamp) == 1:
                            kRamp[-1] = kRamp[-1]/2
                        else:
                            kRamp[-1] = (kRamp[-1]+kRamp[-2])/2
                            kInc = kRamp[-1] - kRamp[-2]
                        logger.warning(f"{kRamp[-1]} because solve failed")
                        # reset fmixed and reinit solver
                        fmixed.vector()[:] = fmixed_prev
                        if custom_solver:
                            problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
                        else:
                            solver = init_solver(Fvar, fmixed, bcs, Jvar)
                    else:
                        logger.warning(f"Resetting pRamp from {pRamp[-1]} to")
                        if len(pRamp) == 1:
                            pRamp[-1] = pRamp[-1]/2
                        else:
                            pRamp[-1] = (pRamp[-1]+pRamp[-2])/2
                            dpress = pRamp[-1] - pRamp[-2]
                        logger.debug(f"{pRamp[-1]} because solve failed")
                        # reset fmixed and reinit solver
                        fmixed.vector()[:] = fmixed_prev
                        if custom_solver:
                            problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
                        else:
                            solver = init_solver(Fvar, fmixed, bcs, Jvar)
                else:
                    if not pumpedUp:
                        keepSwimming = False

            if it >= 100:
                logger.info(f"Done computing idx = {idx} after maximum ({it}) iterations, def = {uEval(0,0,2*nucRad2)[2]}")
                break
            fmixed_prev = fmixed.vector()[:].copy()
            ulin.assign(project(fmixed.sub(0), V_vector))
            testJacobian = project(det(Identity(3) + grad(ulin)), V_scalar)
            print(f"Current min Jacobian is {min(testJacobian.vector())}")

            reinit, keepSwimming, mf_dirichlet, domain_id, uxFixed, uyFixed = update_bcs(
                        fmixed, mf_dirichlet, domain_id, mf_surf, nanopillar_specs, uxFixed, uyFixed)
            if not pumpedUp:
                keepSwimming = False

            if pumpedUp:
                if not inCell:
                    # update pressure
                    linFcn = interpolate(fmixed.sub(0), V1lin)
                    try:
                        uLagrange, volLagrange = lagrangeMap(mf3, 2, mf2, [10,12], 
                                                    [linFcn,linFcn], 1/4, uLagrange)
                    except:
                        logger.warning("Volume could not be computed")
                else:
                    raise ValueError("This case should not be reached here")

                # pressIts.append(pressIts[0] - innerBulkMod*(volLagrange-vol_inner[-1])/volLagrange)
                mAnderson = 3
                mCur = min([mAnderson,len(volEstIts)-1])
                maxPressDev = min([pressIts[-1], 2.0])
                if len(volEstIts)==1:
                    if pressIts[0] == pMax:
                        # then by definition
                        volEstIts[0] = vol0_nuc - vol_ref
                    volIts = [volLagrange] # evaluated from sim
                    gIts = [volLagrange-volEstIts[0]]
                    pTest = computePress(volLagrange)
                    if pTest-pressIts[-1] > maxPressDev:
                        volCur = findVolFromPress(pressIts[-1]+maxPressDev, vol_inner[-1])
                    elif pTest-pressIts[-1] < -maxPressDev:
                        volCur = findVolFromPress(pressIts[-1]-maxPressDev, vol_inner[-1])
                    else:
                        volCur = volLagrange
                    volEstIts.append(volCur)
                else:
                    volIts.append(volLagrange)
                    gIts.append(volLagrange-volEstIts[-1])
                    XCur = np.array(volEstIts[-mCur:]) - np.array(volEstIts[-mCur-1:-1])
                    GCur = np.array(volIts[-mCur:]) - np.array(volIts[-mCur-1:-1])
                    gammaOut = lsq_linear(GCur, gIts[-1])
                    volNucEstNew = volEstIts[-1] + gIts[-1] - np.matmul(XCur + GCur, gammaOut.x)
                    alphaVals = np.diff(gammaOut.x)
                    alphaVals = np.concatenate(([gammaOut.x[0]], alphaVals, [1-gammaOut.x[-1]]))
                    pTest = computePress(volNucEstNew)
                    if pTest-pressIts[-1] > maxPressDev:
                        volNucEstNew = findVolFromPress(pressIts[-1]+maxPressDev, vol_inner[-1])
                    elif pTest-pressIts[-1] < -maxPressDev:
                        volNucEstNew = findVolFromPress(pressIts[-1]-maxPressDev, vol_inner[-1])
                    volEstIts.append(volNucEstNew)
                # pressIts = [(1-fracOsmotic)*pMax + nucSoluteFactor/(volNucEstNew-vol_excl_nuc) - cytoSoluteFactor/(cytoVol - vol_excl_cyto)]
                pressIts.append(computePress(volEstIts[-1]))
                pRamp[-1] = pressIts[-1]
                if np.abs((volIts[-1]-volEstIts[-2])/volIts[-1]) > 0.001:
                    keepSwimming = True 
                alwaysReinit = True
                if alwaysReinit:
                    reinit = True

            if reinit: # then solver needs to be updated
                zDisplFloorExpr = Expression("-z0-hNP-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", 
                                            degree=1, hNP=hNP, r1=nucRad1, r2=nucRad2, z0=z0)
                zDisplFloor = interpolate(zDisplFloorExpr, V)
                bc_nanopillar_z = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
                bc_nanopillar_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 1)
                bc_nanopillar_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 1)
                bc_side_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 8)
                bc_side_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 8)
                bc_floor = DirichletBC(V1.sub(2), zDisplFloor, mf_dirichlet, 7)
                if pumpedUp:
                    bc_symm3_z = DirichletBC(V1.sub(2), Constant(zClampCur), mf_dirichlet2, 8)
                if stick:
                    bcs = [bc_nanopillar_x, bc_nanopillar_y, bc_nanopillar_z, 
                        bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
                else:
                    bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z]
                
                if custom_solver:
                    problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
                else:
                    solver = init_solver(Fvar, fmixed, bcs, Jvar)
            
            if keepSwimming:
                logger.info(f"Computing idx = {idx}, outer iteration {it}, def = {uEval(0,0,2*nucRad2)[2]}")
            else:
                logger.info(f"Done computing idx = {idx} after {it} iterations, def = {uEval(0,0,2*nucRad2)[2]}")

        zTop = zShift + 2*nucRad2 + uEval(0,0,2*nucRad2)[2]
        zTops.append(zTop)
        if zTop < zRoofPM-dSteric and pumpedUp:
            raise ValueError("Full cell not supported in this version")
        
        # set next force values
        linFcn = interpolate(fmixed.sub(0), V1lin)
        uLagrange, volLagrange = lagrangeMap(mf3, 2, mf2, [10,12], 
                                    [linFcn,linFcn], 1/4, uLagrange)

        vol_inner.append(volLagrange)
        
        if pRamp[-1] >= pMax and not pumpedUp:
            pumpedUp = True
            kRamp.append(kMin)
            pRamp.append(pMax)
            zDisplFloorExpr = Expression("-hNP-z0-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", 
                                            degree=1, hNP=hNP, r1=nucRad1, r2=nucRad2, z0=z0)
            zDisplFloor = interpolate(zDisplFloorExpr, V)
            mf_dirichlet2.array()[mf_dirichlet2.array()==8] = 0 # remove pressure symm condition
            for f in facets(mesh):
                if mf_surf[f] == 10: # on outer surface
                    for v in vertices(f):
                        if v.x(0)<1e-6 and v.x(1)<1e-6 and v.x(2) > z0+nucRad2:
                            mf_surfInt[f] = 0
                            mf_dirichlet2[f] = 8
                            break
            zClampCur = (assemble(uEval[2]*sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1))/
                         assemble(sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1)))#uEval(0,npRad,z0)[2]
            bc_symm3_z = DirichletBC(V1.sub(2), Constant(zClampCur), mf_dirichlet2, 8)
            # bc_symm3_z = DirichletBC(V1.sub(2), Constant(0.0), mf_dirichlet2, 8)
            bc_nanopillar_z = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
            bc_nanopillar_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 1)
            bc_nanopillar_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 1)
            bc_side_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 8)
            bc_side_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 8)
            bc_floor = DirichletBC(V1.sub(2), zDisplFloor, mf_dirichlet, 7)
            if stick:
                bcs = [bc_nanopillar_x, bc_nanopillar_y, bc_nanopillar_z, 
                    bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
            else:
                bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z]
            if custom_solver:
                problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
            else:
                solver = init_solver(Fvar, fmixed, bcs, Jvar)
            volEstIts = [vol_inner[-1]]
        elif pumpedUp:
            zClampCur = (assemble(uEval[2]*sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1))/
                                    assemble(sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1)))
            kRamp.append(min([kRamp[-1]+kInc, kMax]))
            volNucEst = vol_inner[-1]
            volEstIts = [volNucEst]
            pNew = computePress(volNucEst)
            pRamp.append(pNew)
            if kRamp[-1] >= kMax:
                stopLogic = True # then done with this simulation
        else:
            pRamp.append(min([pRamp[-1]+dpress, pMax]))
            kRamp.append(kMin)   
        u_file.write_checkpoint(fmixed.sub(0), "u_np_ellipsoid", idx, append=True)
        ulin_file.write(ulin, idx)

        if len(fmixed.sub(0).vector()) == len(u_prev.vector()):
            u_prev.vector()[:] = fmixed.sub(0).vector()[:].copy()
            u_prev.vector().apply("insert")
        else:
            raise ValueError("Could not reassign u_prev!!")

        if len(psi_c.vector()) == len(psi_c_prev.vector()):
            psi_c_prev.vector()[:] = psi_c.vector()[:].copy()
            psi_c_prev.vector().apply("insert")
        else:
            raise ValueError("Could not reassign psi_c_prev!!")

        a_vector_new = J * dot(normals, inv(F))
        a_vector_new = project(a_vector_new, V_vector)
        a_vector.assign(a_vector_new)
        a_file.write(a_vector, idx)

        # save stress and strain energy density for current configuration
        surf_stress.assign(project(dot(normals, Ttensor)/sqrt(inner(outer_area_factor,outer_area_factor)), V_vector))
        surf_stress_file.write(surf_stress, idx)
        TCtensor = dot(Ttensor, F.T) / J # cauchy stress tensor
        von_mises_calc = sqrt((TCtensor[0,0] - TCtensor[1,1])**2 + (TCtensor[1,1] - TCtensor[2,2])**2 + (TCtensor[2,2] - TCtensor[0,0])**2
                    + 6*(TCtensor[0,1]**2 + TCtensor[0,2]**2 + TCtensor[2,1]**2))/np.sqrt(2)
        von_mises.assign(project(von_mises_calc, V_scalar))
        vonmises_stress_file.write(von_mises, idx)
        psi_fcn.assign(project(psi, V_scalar))
        psi_file.write(psi_fcn, idx)

        surf_tension.assign(project(dot(dot(thetas, Ttensor), thetas)/sqrt(inner(theta_area_factor,theta_area_factor)) + 
                           dot(dot(phis, Ttensor), phis)/sqrt(inner(phi_area_factor,phi_area_factor)), V_scalar))
        surf_tension_file.write(surf_tension, idx)

        # save relevant vol and SAs
        inner_SA.append(assemble(a_scalar*ds_integrate(12))/inner_SA_ref)
        outer_SA.append(assemble(a_scalar*ds_integrate(10))/outer_SA_ref)
        vol.append(assemble(J*dx))#/vol_ref)

        np.savetxt(f"{results_folder}/inner_SA.txt", inner_SA)
        np.savetxt(f"{results_folder}/outer_SA.txt", outer_SA)
        np.savetxt(f"{results_folder}/vol.txt", vol)
        np.savetxt(f"{results_folder}/vol_inner.txt", vol_inner)
        np.savetxt(f"{results_folder}/pRamp.txt", pRamp)
        np.savetxt(f"{results_folder}/kRamp.txt", kRamp)
        np.savetxt(f"{results_folder}/zTops.txt", zTops)

        if inCell or stopLogic:
            break

    nuc_dict = {"idx": idx, "pRamp": pRamp, "kRamp": kRamp, "forceConst": forceConst, "kInc": kInc,
                "pressConst": pressConst, "fmixed": fmixed, "solver": solver, "Fvar": Fvar,
                "Jvar": Jvar, "bcs": bcs, "mf_dirichlet": mf_dirichlet, "mf_dirichlet2": mf_dirichlet2, "domain_id": domain_id, 
                "mf_surf": mf_surf, "nanopillar_specs": nanopillar_specs, "uxFixed": uxFixed, "uyFixed": uyFixed,
                "vol_inner": vol_inner, "nucRad1": nucRad1, "nucRad2": nucRad2, "zDisplOuter": zDisplOuter, 
                "dSteric": dSteric, "u_prev": u_prev, "a_vector": a_vector, "u_file": u_file, "a_file": a_file, 
                "results_folder": results_folder, "inner_SA": inner_SA, "outer_SA": outer_SA, "vol": vol,
                "volLagrange": volLagrange, "rthickness": rthickness, "zthickness": zthickness, 
                "fmixed_prev": fmixed_prev, "mf3": mf3, "mf2": mf2, "uLagrange": uLagrange,
                "innerBulkMod": innerBulkMod, "ulin": ulin, "ulin_file": ulin_file, "zTops": zTops, "nuc_only": nuc_only,
                "surf_stress": surf_stress, "surf_stress_file": surf_stress_file, "von_mises": von_mises, 
                "vonmises_stress_file": vonmises_stress_file, "psi_fcn": psi_fcn, "psi_file": psi_file,
                "surf_tension": surf_tension, "surf_tension_file": surf_tension_file,
                "E1scale": E1scale, "E2scale": E2scale, "psi_c_prev": psi_c_prev, "p0": p0, "pressBoost": pressBoost,
                "ds_roofInt": ds_roofInt, "npContactForce": npContactForce, "softFactor": args["softFactor"]}
    
    return nuc_dict

def init_solver(Fvar, fmixed, bcs, Jvar):
    problem = NonlinearVariationalProblem(Fvar, fmixed, bcs, J=Jvar)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-6
    prm["newton_solver"]["relative_tolerance"] = 1E-6
    prm["newton_solver"]["maximum_iterations"] = 100
    prm["newton_solver"]["linear_solver"] = 'mumps'
    return solver
        
class SNESProblem():
    def __init__(self, F, fmixed, u, p, bcs):
        V = fmixed.function_space()
        # (du, dp) = TrialFunctions(V)
        self.L = F
        # self.a = expand_derivatives(derivative(F, u) + 
        #           derivative(F, p))
        # dw = TrialFunction(V)
        self.a = derivative(F, fmixed)#, dw)
        self.bcs = bcs
        self.u = u
        self.p = p
        self.fmixed = fmixed

    def F(self, snes, x, F):
        x = PETScVector(x)
        F  = PETScVector(F)
        assemble(self.L, tensor=F)
        for bc in self.bcs:
            bc.apply(F, x)                

    def J(self, snes, x, J, P):
        J = PETScMatrix(J)
        assemble(self.a, tensor=J)#, keep_diagonal=True)
        # test_vals = as_backend_type(J)
        for bc in self.bcs:
            bc.apply(J)

def init_custom_solver(Fvar, fmixed, u, p, bcs):     
    problem = SNESProblem(Fvar, fmixed, u, p, bcs)
        
    b = PETScVector()  # same as b = PETSc.Vec()
    J_mat = PETScMatrix()

    solver = PETSc.SNES().create(MPI.comm_world)    
    solver.setFunction(problem.F, b.vec())
    solver.setJacobian(problem.J, J_mat.mat())
    solver.setType("newtonls")
    solver.setTolerances(rtol=1e-5)

    def monitor(snes, it, fgnorm):
        # prints out residual at each Newton iteration
        print("  " + str(it) + " SNES Function norm " + "{:e}".format(fgnorm))

    solver.setMonitor(monitor)
    solver.ksp.setType("gmres")
    solver.ksp.setTolerances(rtol=1e-5)
    solver.ksp.pc.setType("none")
    opts = PETSc.Options()
    opts["snes_linesearch_type"] = "basic"
    solver.setFromOptions()
    return (problem, solver)

# define Lagrangian mapping for nuclear interior and/or cytosol
def lagrangeMap(mf3, domain_id, mf2, bound_ids, uBounds, symm, u0, bulk_mod = 1e5, degree=1):
    if isinstance(u0, Function):
        V = u0.function_space()
        mesh = V.mesh()
        udef = u0
    else:
        mesh = mf3.mesh()#create_meshview(mf3, domain_id)
        V = VectorFunctionSpace(mesh, "P", degree)
        # Vscalar = FunctionSpace(mesh, "P", 1)
        udef = Function(V)
    dudef = TrialFunction(V)
    v = TestFunction(V)

    # Kinematics
    dim = 3
    I = Identity(dim)             # Identity tensor

    # global kinematics
    F = I + grad(udef)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor
    J  = det(F)
    I1 = tr(C)# * J**(-2/3)
    I2 = 0.5*(I1**2 - tr(C*C))# * J**(-4/3)

    # Elasticity parameters
    E1_nominal = 5000.0
    E2_nominal = 0.0#5000.0


    # markers for symmetry conditions
    mf_bc = mf2 #MeshFunction("size_t", mesh, 2, 0)
    class SymmAxis1(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] < 0.001 and on_boundary
    # if symm == 1/8:
    #     class SymmAxis2(SubDomain):
    #         def inside(self, x, on_boundary):
    #             return (np.arctan2(x[1],x[0]) > 0.99*np.pi/4 or 
    #                     (x[1]==0 and x[0]==0)) and on_boundary
    
    symm1 = SymmAxis1()
    symm1.mark(mf_bc, 2)
    if symm == 1/4:
        class SymmAxis2(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.001 and on_boundary
        symm2 = SymmAxis2()
        symm2.mark(mf_bc, 3)
    elif symm == 1/8:
        class AllBound(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary
        allbound = AllBound()
        mf_allbound = MeshFunction("size_t", mesh, 2)
        allbound.mark(mf_allbound, 1)
        for f in facets(mesh):
            if mf_bc[f] == 0 and mf_allbound[f] == 1:
                mf_bc[f] = 3 # any leftover boundaries belong to 45 deg angle
    else:
        raise ValueError("Symmetry must be either 1/8 or 1/4")
    ds_global = Measure("ds", domain=mesh, subdomain_data=mf_bc)
    bcs = []
    for i in range(len(uBounds)):
        if uBounds[i].function_space().mesh().id() != mesh.id():
            uBound = sub_to_parent(uBounds[i], mesh)
        else:
            uBound = uBounds[i]
        bcs.append(DirichletBC(V, uBound, mf_bc, bound_ids[i]))
    
    dx_global = Measure("dx", domain=mesh, subdomain_data=mf3)

    # Stored strain energy density (neo-Hookean)
    Pi_global = (E1_nominal*(I1-dim) + E2_nominal*(I2-dim) + bulk_mod*(J-1)**2)*dx_global

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    Fvar = derivative(Pi_global, udef, v)

    # apply penalties for violations of symmetry (if needed)
    if symm == 1/8:
        penVal = 1e3
        Fvar = Fvar + penVal*(udef[1])*v[1]*ds_global(2) + penVal*(udef[0]-udef[1])*(v[0]-v[1])*ds_global(3)
    elif symm == 1/4:
        bcs.append(DirichletBC(V.sub(1), Constant(0.0), mf_bc, 2))
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), mf_bc, 3))

    # Compute Jacobian of F
    Jvar = derivative(Fvar, udef, dudef)

    # Solve variational problem
    problem = NonlinearVariationalProblem(Fvar, udef, bcs, J=Jvar)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-8
    prm["newton_solver"]["relative_tolerance"] = 1E-6
    prm["newton_solver"]["maximum_iterations"] = 100
    prm["newton_solver"]["linear_solver"] = 'mumps'
    solver.solve()

    if symm == 1/8:
        mesh = mf3.mesh()#create_meshview(mf3, domain_id)
        Vlin = VectorFunctionSpace(mesh, "P", 1)
        udef = interpolate(udef, Vlin)
        # then correct deviations from symmetry
        coords = Vlin.tabulate_dof_coordinates()
        uvec = udef.vector()[:]
        for i in range(0,len(coords),3):
            if coords[i,0] < 0.001:
                uvec[i+1] = 0.0
            elif np.arctan2(coords[i,1],coords[i,0]) > 0.999*np.pi/4:
                avg_xy = (uvec[i]+uvec[i+1])/2
                uvec[i] = avg_xy
                uvec[i+1] = avg_xy
        udef.vector().set_local(uvec)
        udef.vector().apply("insert")

    ref_vol = assemble(1.0*dx_global(domain_id))
    vol = assemble(J*dx_global(domain_id))
    logger.info(f'Current volume is {vol} (ref is {ref_vol})')

    return (udef, vol)

def solve_next_nuc_step(nuc_dict):
    idx = nuc_dict["idx"]
    pRamp = nuc_dict["pRamp"]
    kRamp = nuc_dict["kRamp"]
    forceConst = nuc_dict["forceConst"]
    kInc = nuc_dict["kInc"]
    pressConst = nuc_dict["pressConst"]
    fmixed = nuc_dict["fmixed"]
    solver = nuc_dict["solver"]
    Fvar = nuc_dict["Fvar"]
    Jvar = nuc_dict["Jvar"]
    bcs = nuc_dict["bcs"]
    mf_dirichlet = nuc_dict["mf_dirichlet"]
    mf_dirichlet2 = nuc_dict["mf_dirichlet2"]
    domain_id = nuc_dict["domain_id"] 
    mf_surf = nuc_dict["mf_surf"]
    nanopillar_specs = nuc_dict["nanopillar_specs"] 
    uxFixed = nuc_dict["uxFixed"] 
    uyFixed = nuc_dict["uyFixed"]
    vol_inner = nuc_dict["vol_inner"]
    nucRad1 = nuc_dict["nucRad1"]
    nucRad2 = nuc_dict["nucRad2"]
    zDisplOuter = nuc_dict["zDisplOuter"]
    dSteric = nuc_dict["dSteric"]
    u_prev = nuc_dict["u_prev"]
    a_vector = nuc_dict["a_vector"]
    u_file = nuc_dict["u_file"]
    a_file = nuc_dict["a_file"]
    results_folder = nuc_dict["results_folder"]
    inner_SA = nuc_dict["inner_SA"]
    outer_SA = nuc_dict["outer_SA"]
    vol = nuc_dict["vol"]
    volLagrange = nuc_dict["volLagrange"]
    rthickness = nuc_dict["rthickness"]
    zthickness = nuc_dict["zthickness"]
    fmixed_prev = nuc_dict["fmixed_prev"]
    mf3 = nuc_dict["mf3"]
    mf2 = nuc_dict["mf2"]
    uLagrange = nuc_dict["uLagrange"]
    innerBulkMod = nuc_dict["innerBulkMod"]
    ulin = nuc_dict["ulin"]
    ulin_file = nuc_dict["ulin_file"]
    zTops = nuc_dict["zTops"]
    surf_stress = nuc_dict["surf_stress"]
    surf_stress_file = nuc_dict["surf_stress_file"]
    von_mises = nuc_dict["von_mises"]
    vonmises_stress_file = nuc_dict["vonmises_stress_file"]
    psi_fcn = nuc_dict["psi_fcn"]
    psi_file = nuc_dict["psi_file"]
    E1scale = nuc_dict["E1scale"]
    E2scale = nuc_dict["E2scale"]
    psi_c_prev = nuc_dict["psi_c_prev"]
    ds_roofInt = nuc_dict["ds_roofInt"]
    npContactForce = nuc_dict["npContactForce"]
    surf_tension = nuc_dict["surf_tension"]
    surf_tension_file = nuc_dict["surf_tension_file"]
    softFactor = nuc_dict["softFactor"]
    p0 = nuc_dict["p0"]
    pressBoost = nuc_dict["pressBoost"]
    stick = False

    if not nuc_dict["nuc_only"]:
        raise ValueError("Full cell simulations not supported in this version")

    mesh = fmixed.function_space().mesh()
    x = SpatialCoordinate(mesh)
    (u, p) = split(fmixed)
    I = variable(Identity(3))             # Identity tensor
    F = variable(I + grad(u))             # Deformation gradient
    J  = variable(det(F))
    a_scalar = ufl.sqrt(a_vector[0]**2 + a_vector[1]**2 + a_vector[2]**2)
    ds_integrate = Measure('ds', domain=mesh, subdomain_data=mf_surf)
    dx_integrate = Measure('dx', domain=mesh)
    inner_SA_ref = assemble(1.0*ds_integrate(12))
    outer_SA_ref = assemble(1.0*ds_integrate(10))
    vol_ref = assemble(1.0*dx_integrate)

    hNP = nanopillar_specs["hNP"]

    zShift = hNP + dSteric
    V1 = fmixed.function_space().sub(0)
    bc_symm1_y = DirichletBC(V1.sub(1), Constant(0.0), mf_dirichlet, 2)
    bc_symm2_x = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 3)
    zRoofPM = 0
    
    V_vector = FunctionSpace(mesh, VectorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
    V = FunctionSpace(mesh, "P", 1)
    z0 = min(mesh.coordinates()[:,2])
    normal_expr = Expression(("x[0]/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))",
                              "x[1]/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))",
                              "(x[2]-z0)/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))"),
                            z0=nucRad2+z0, degree=1)
    theta_expr = Expression(("-x[1]/sqrt(pow(x[0],2) + pow(x[1],2) + 0.0001)",
                            "x[0]/sqrt(pow(x[0],2) + pow(x[1],2) + 0.0001)",
                            "0"), degree=1)
    normals = project(normal_expr, V_vector)
    thetas = project(theta_expr, V_vector)
    theta_coords = V_vector.tabulate_dof_coordinates()
    theta_vec = thetas.vector()[:]
    for i in range(0,len(theta_vec),3):
        if theta_coords[i][0] < 1e-6 and theta_coords[i][1] < 1e-6:
            theta_vec[i:i+3] = [0.,1.,0.]
    thetas.vector().set_local(theta_vec)
    thetas.vector().apply("insert")
    phis = cross(normals,thetas)
    outer_area_factor = J * dot(normals, inv(F))
    theta_area_factor = J * dot(phis, inv(F))
    phi_area_factor = J * dot(thetas, inv(F))

    uEval = fmixed.sub(0)
    uEval.set_allow_extrapolation(True)

    idx += 1
    keepSwimming = True
    it = 0
    vol0_nuc = 142.0 + vol_ref
    vol_tot = vol0_nuc/0.3627 #1024.0
    vol0_cyto = vol_tot - vol0_nuc
    vol_excl_nuc = (vol0_nuc-vol_ref)*0.01
    vol_excl_cyto = vol0_cyto*0.02
    pMax = 820.0 * softFactor
    baselineOsmotic = 2400.0
    fracOsmotic = (pMax-pressBoost)/pMax
    nucSoluteFactor = (fracOsmotic*pMax + baselineOsmotic) * (vol0_nuc - vol_ref - vol_excl_nuc) 
    nucBoostFactor = (pressBoost) * (vol0_nuc - vol_ref)**2
    cytoSoluteFactor = (baselineOsmotic) * (vol0_cyto - vol_excl_cyto)
    def computeBothPress(nucVol):
        cytoVol = vol_tot - nucVol - vol_ref
        pNuc = nucSoluteFactor/(nucVol-vol_excl_nuc) + nucBoostFactor/(nucVol**2)
        pCyto = cytoSoluteFactor/(cytoVol - vol_excl_cyto)
        return pNuc, pCyto
    def computePress(nucVol):
        cytoVol = vol_tot - nucVol - vol_ref
        pTest = (nucSoluteFactor/(nucVol-vol_excl_nuc) + nucBoostFactor/(nucVol**2) - 
                    cytoSoluteFactor/(cytoVol - vol_excl_cyto))
        return pTest
    def rootPress(nucVol, pressVal):
        return computePress(nucVol) - pressVal
    def findVolFromPress(pressVal, x0):
        bracketVals=[0.5*x0, 1.5*x0]
        pressAns = root_scalar(rootPress, args=(pressVal,), bracket=bracketVals)
        return pressAns.root
    # if kRamp[-2] == 0:
    volEstIts = [vol_inner[-1]]
    # else:
    #     volSlope = (vol_inner[-1]-vol_inner[-2])/(kRamp[-2]-kRamp[-3])
    #     volEstIts = [vol_inner[-1] + (kRamp[-1]-kRamp[-2])*volSlope]
    pressIts = [computePress(volEstIts[-1])]
    volIts = []
    pRamp[-1] = pressIts[-1]
    zClampCur = (assemble(uEval[2]*sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1))/
                    assemble(sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1)))#uEval(0,npRad,z0)[2]
    while keepSwimming:
        # need to keep running the solver until the BCs do not change for the current force
        it += 1
        set_step = False
        # try solving, if diverges, take smaller step
        while not set_step:
            curPressNuc, curPressCyto = computeBothPress(volEstIts[-1])
            assert curPressNuc - curPressCyto == pRamp[-1], "Pressure mismatch"
            curForce = kRamp[-1]
            logger.info(f"Current force is {curForce}")
            logger.info(f"Current pressure is {pRamp[-1]}")
            forceConst.assign(curForce)
            pressConst.assign(curPressNuc-curPressCyto)
            p0.assign(curPressCyto)
            reset = False
            try:
                solver.solve()
                set_step = True
            except:
                reset = True
            if reset:
                logger.warning(f"Resetting kRamp from {kRamp[-1]} to")
                if len(kRamp) == 1:
                    kRamp[-1] = kRamp[-1]/2
                else:
                    kRamp[-1] = (kRamp[-1]+kRamp[-2])/2
                    kInc = kRamp[-1] - kRamp[-2]
                logger.warning(f"{kRamp[-1]} because solve failed")
                # reset fmixed and reinit solver
                fmixed.vector()[:] = fmixed_prev
                solver = init_solver(Fvar, fmixed, bcs, Jvar)
                volEstIts = [vol_inner[-1]]
                pressIts = [computePress(volEstIts[-1])]
                volIts = []
                pRamp[-1] = pressIts[-1]
                it = 1

        if it >= 100:
            logger.info(f"Done computing idx = {idx} after maximum ({it}) iterations, def = {uEval(0,0,2*nucRad2)[2]}")
            break
        fmixed_prev = fmixed.vector()[:].copy()
        
        reinit, keepSwimming, mf_dirichlet, domain_id, uxFixed, uyFixed = update_bcs(
            fmixed, mf_dirichlet, domain_id, mf_surf, nanopillar_specs, uxFixed, uyFixed)
        # update pressure
        V1lin = VectorFunctionSpace(mesh, "P", 1)
        linFcn = interpolate(fmixed.sub(0), V1lin)
        uLagrange, volLagrange = lagrangeMap(mf3, 2, mf2, [10,12], 
                                    [linFcn,linFcn], 1/4, uLagrange)
        # pressIts.append(pressIts[0] - innerBulkMod*(volLagrange-vol_inner[-1])/volLagrange)
        mAnderson = 3
        mCur = min([mAnderson,len(volEstIts)-1])
        maxPressDev = min([pressIts[-1], 2.0])
        if len(volEstIts)==1:
            volIts = [volLagrange] # evaluated from sim
            gIts = [volLagrange-volEstIts[0]]
            pTest = computePress(volLagrange)
            if pTest-pressIts[-1] > maxPressDev:
                volCur = findVolFromPress(pressIts[-1]+maxPressDev, vol_inner[-1])
            elif pTest-pressIts[-1] < -maxPressDev:
                volCur = findVolFromPress(pressIts[-1]-maxPressDev, vol_inner[-1])
            else:
                volCur = volLagrange
            volEstIts.append(volCur)
        else:
            volIts.append(volLagrange)
            gIts.append(volLagrange-volEstIts[-1])
            XCur = np.array(volEstIts[-mCur:]) - np.array(volEstIts[-mCur-1:-1])
            GCur = np.array(volIts[-mCur:]) - np.array(volIts[-mCur-1:-1])
            gammaOut = lsq_linear(GCur, gIts[-1])
            volNucEstNew = volEstIts[-1] + gIts[-1] - np.matmul(XCur + GCur, gammaOut.x)
            alphaVals = np.diff(gammaOut.x)
            alphaVals = np.concatenate(([gammaOut.x[0]], alphaVals, [1-gammaOut.x[-1]]))
            pTest = computePress(volNucEstNew)
            if pTest-pressIts[-1] > maxPressDev:
                volNucEstNew = findVolFromPress(pressIts[-1]+maxPressDev, vol_inner[-1])
            elif pTest-pressIts[-1] < -maxPressDev:
                volNucEstNew = findVolFromPress(pressIts[-1]-maxPressDev, vol_inner[-1])
            volEstIts.append(volNucEstNew)
        # pressIts.append(0.5*pMax + nucSoluteFactor/(volNucEst-vol_excl_nuc) - cytoSoluteFactor/(cytoVol - vol_excl_cyto))
        pressIts.append(computePress(volEstIts[-1]))
        pRamp[-1] = pressIts[-1]
        if np.abs((volIts[-1]-volEstIts[-2])/volIts[-1]) > 0.001:
            keepSwimming = True 
        
        alwaysReinit = True
        if alwaysReinit:
            reinit = True

        if reinit: # then solver needs to be updated
            zDisplFloorExpr = Expression("-hNP-z0-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", 
                                        degree=1, hNP=hNP, r1=nucRad1, r2=nucRad2, z0=z0)
            zDisplFloor = interpolate(zDisplFloorExpr, V)
            bc_nanopillar_z = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
            bc_nanopillar_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 1)
            bc_nanopillar_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 1)
            bc_side_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 8)
            bc_side_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 8)
            bc_floor = DirichletBC(V1.sub(2), zDisplFloor, mf_dirichlet, 7)
            bc_symm3_z = DirichletBC(V1.sub(2), Constant(zClampCur), mf_dirichlet2, 8)
            if stick:
                bcs = [bc_nanopillar_x, bc_nanopillar_y, bc_nanopillar_z, bc_symm3_z,
                        bc_floor, bc_symm1_y, bc_symm2_x, bc_side_x, bc_side_y]
            else:
                # bcs = [bc_nanopillar_z, bc_floor, bc_symm1_y, bc_symm2_x, bc_side_x, bc_side_y]
                bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z]
            solver = init_solver(Fvar, fmixed, bcs, Jvar)
        
        if keepSwimming:
            logger.info(f"Computing idx = {idx}, outer iteration {it}, def = {uEval(0,0,2*nucRad2)[2]}")
        else:
            logger.info(f"Done computing idx = {idx} after {it} iterations, def = {uEval(0,0,2*nucRad2)[2]}")
    zTop = zShift + 2*nucRad2 + uEval(0,0,2*nucRad2)[2]
    zTops.append(zTop)
    if zTop < zRoofPM-dSteric:
        raise ValueError("Full cell simulations not supported in this version")
        
    # set next pressure values
    linFcn = interpolate(fmixed.sub(0), V1lin)
    try:
        uLagrange, volLagrange = lagrangeMap(mf3, 2, mf2, [10,12], 
                                    [linFcn,linFcn], 1/4, uLagrange)
    except:
        logger.warning("Current volume could not be solved for")
    vol_inner.append(volLagrange)
    # pNew = pRamp[-1] - innerBulkMod*(vol_inner[-1]-vol_inner[-2])/vol_inner[-1]
    # pNew = 0.5*pMax + nucSoluteFactor/(volLagrange-vol_excl_nuc) - cytoSoluteFactor/(cytoVol - vol_excl_cyto)
    pNew = computePress(volEstIts[-1])

    u_file.write_checkpoint(fmixed.sub(0), "u_np_ellipsoid", idx, append=True)
    ulin.assign(project(fmixed.sub(0), V_vector))
    ulin_file.write(ulin, idx)

    if len(fmixed.sub(0).vector()) == len(u_prev.vector()):
        u_prev.vector()[:] = fmixed.sub(0).vector()[:]
        u_prev.vector().apply("insert")
    else:
        raise ValueError("Could not reassign u_prev!!")

    # compute stretch for current configuration
    a_vector_new = J * dot(normals, inv(F))
    a_vector_new = project(a_vector_new, V_vector)
    a_vector.assign(a_vector_new)
    a_file.write(a_vector, idx)

    # save stress and strain energy density for current configuration
    I = variable(Identity(3))             # Identity tensor
    F = variable(I + grad(u))             # Deformation gradient
    C = variable(F.T*F)                   # Right Cauchy-Green tensor
    (u, p) = fmixed.split()
    E1, E2 = Constant(5000.0), Constant(1000.0)
    I1 = variable(tr(C))
    I2 = variable(0.5*(I1**2 - tr(C*C)))
    J  = variable(det(F))
    psi = E1*(I1-3) + E2*(I2-3) - p*(J-1)
    Ttensor = diff(psi, F)
    V_scalar = FunctionSpace(mesh, "P", 1)
    surf_stress.assign(project(dot(normals, Ttensor)/sqrt(inner(outer_area_factor,outer_area_factor)), V_vector))
    surf_stress_file.write(surf_stress, idx)
    TCtensor = dot(Ttensor, F.T) / J # cauchy stress tensor
    von_mises_calc = sqrt((TCtensor[0,0] - TCtensor[1,1])**2 + (TCtensor[1,1] - TCtensor[2,2])**2 + (TCtensor[2,2] - TCtensor[0,0])**2
                    + 6*(TCtensor[0,1]**2 + TCtensor[0,2]**2 + TCtensor[2,1]**2))/np.sqrt(2)
    von_mises.assign(project(von_mises_calc, V_scalar))
    vonmises_stress_file.write(von_mises, idx)
    psi_fcn.assign(project(psi, V_scalar))
    psi_file.write(psi_fcn, idx)
    surf_tension.assign(project(dot(dot(thetas, Ttensor), thetas)/sqrt(inner(theta_area_factor,theta_area_factor)) + 
                           dot(dot(phis, Ttensor), phis)/sqrt(inner(phi_area_factor,phi_area_factor)), V_scalar))
    surf_tension_file.write(surf_tension, idx)

    # save relevant vol and SAs
    inner_SA.append(assemble(a_scalar*ds_integrate(12))/inner_SA_ref)
    outer_SA.append(assemble(a_scalar*ds_integrate(10))/outer_SA_ref)
    vol.append(assemble(J*dx))#/vol_ref)
    kRamp.append(kRamp[-1]+kInc)
    pRamp.append(pNew)

    np.savetxt(f"{results_folder}/inner_SA.txt", inner_SA)
    np.savetxt(f"{results_folder}/outer_SA.txt", outer_SA)
    np.savetxt(f"{results_folder}/vol.txt", vol)
    np.savetxt(f"{results_folder}/vol_inner.txt", vol_inner)
    np.savetxt(f"{results_folder}/pRamp.txt", pRamp)
    np.savetxt(f"{results_folder}/kRamp.txt", kRamp)
    np.savetxt(f"{results_folder}/zTops.txt", zTops)
    nuc_dict = {"idx": idx, "pRamp": pRamp, "kRamp": kRamp, "forceConst": forceConst, "kInc": kInc,
                "pressConst": pressConst, "fmixed": fmixed, "solver": solver, "Fvar": Fvar,
                "Jvar": Jvar, "bcs": bcs, "mf_dirichlet": mf_dirichlet, "mf_dirichlet2": mf_dirichlet2, "domain_id": domain_id, 
                "mf_surf": mf_surf, "nanopillar_specs": nanopillar_specs, "uxFixed": uxFixed, "uyFixed": uyFixed,
                "vol_inner": vol_inner, "nucRad1": nucRad1, "nucRad2": nucRad2, "zDisplOuter": zDisplOuter, 
                "dSteric": dSteric, "u_prev": u_prev, "a_vector": a_vector, "u_file": u_file, "a_file": a_file, 
                "results_folder": results_folder, "inner_SA": inner_SA, "outer_SA": outer_SA, "vol": vol, 
                "volLagrange": volLagrange, "rthickness": rthickness, "zthickness": zthickness, 
                "fmixed_prev": fmixed_prev, "mf3": mf3, "mf2": mf2, "uLagrange": uLagrange,
                "innerBulkMod": innerBulkMod, "ulin": ulin, "ulin_file": ulin_file, "zTops": zTops, "nuc_only": nuc_dict["nuc_only"],
                "surf_stress": surf_stress, "surf_stress_file": surf_stress_file, "von_mises": von_mises, 
                "vonmises_stress_file": vonmises_stress_file, "psi_fcn": psi_fcn, "psi_file": psi_file,
                "surf_tension": surf_tension, "surf_tension_file": surf_tension_file,
                "E1scale": E1scale, "E2scale": E2scale, "psi_c_prev": psi_c_prev, "p0": p0, "pressBoost": pressBoost,
                "ds_roofInt": ds_roofInt, "npContactForce": npContactForce, "softFactor": softFactor}
    return nuc_dict

def update_bcs(fmixed, mf_dirichlet, domain_id, mf_surf, 
               nanopillar_specs, uxFixed, uyFixed):
    # update bcs as needed
    reinit = False
    keepSwimming = False
    uEval = fmixed.sub(0)
    uEval.set_allow_extrapolation(True)
    # load nanopillar specs
    hNP = nanopillar_specs["hNP"]
    npRad = nanopillar_specs["rNP"]
    xNP = nanopillar_specs["xNP"]
    yNP = nanopillar_specs["yNP"]
    dofmap_x = vertex_to_dof_map(uxFixed.function_space())
    dofmap_y = vertex_to_dof_map(uyFixed.function_space())
    zRoof = np.inf
    mesh = fmixed.function_space().mesh()
    stick = False
    V_vector = VectorFunctionSpace(mesh, "P", 1)
    ulin = project(uEval, V_vector)
    ulin.set_allow_extrapolation(True)

    for f in facets(mesh):
        if mf_surf[f] == 10:
            xCur = [f.midpoint().x(), f.midpoint().y(), f.midpoint().z()]
            uCur = uEval(xCur)
            xDef = xCur + uCur
            all_dist = np.sqrt((xNP-xDef[0])**2 + (yNP-xDef[1])**2)
            if xDef[2] <= 2.0:#1e-6:
                if xDef[2] <= (-hNP+1e-6) and mf_dirichlet[f] != 7:
                    reinit = True
                    keepSwimming = True
                    mf_dirichlet[f] = 7
                    # domain_id[f] = 11
                elif mf_dirichlet[f] == 7 and not (xDef[2] <= (-hNP+1e-6)):
                    reinit = True
                    keepSwimming = True
                    mf_dirichlet[f] = 0
            elif xDef[2] >= zRoof and mf_dirichlet[f] != 4:
                domain_id[f] = 1
                reinit = True
                mf_dirichlet[f] = 4
                keepSwimming = True
        
    if reinit:
        logger.debug("Need to reinit")
    return (reinit, keepSwimming, mf_dirichlet, domain_id, uxFixed, uyFixed)
