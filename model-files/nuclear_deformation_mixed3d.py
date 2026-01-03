from dolfin import *
from smart import mesh_tools
import numpy as np
import pathlib
import sys
import argparse, logging
import petsc4py.PETSc as PETSc

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
        args["start_force"] = 0.5
        args["u0"] = pathlib.Path("")
        args["bulk_mod"] = 1e8
        args["nanopillar_radius"] = 0.0 #0.25#0.5
        args["nanopillar_height"] = 0.0 #3.0
        args["nanopillar_spacing"] = 0.0 #3.5
        args["contactRad"] = 15.5
        args["outdir"] = pathlib.Path(f"/root/scratch/nuc_indent_quickTestLowLamin")#tallerPartialSlip")
        args["nuc_only"] = True
        args["softFactor"] = 0.1
        args["bulk_mod"] *= args["softFactor"]
    
    nuc_only = args["nuc_only"]
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
        hEdge = args["nanopillar_radius"]/2
        if hEdge > 0.15:
            hEdge = 0.15
    zRoof = np.inf
    zRoofAlt = 10.0

    mesh_ref, mf2, mf3 = mesh_gen.NE_mesh(rRad=nucRad1, zRad=nucRad2, thickness=[rthickness,zthickness],
                                        hEdge=hEdge, hInnerEdge=hEdge, sym_fraction=0.25, NE_layers=NE_layers,
                                        use_tmp=True, hTop=1.5*hEdge)
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
    mf_edge = MeshFunction("size_t", mesh, 2, 0)

    # redetermine surface markers
    rad_eff1 = (nucRad1**2 * nucRad2)**(1/3)
    rad_eff2 = ((nucRad1-rthickness)**2 * (nucRad2-zthickness))**(1/3)
    thickness_thresh = ((rad_eff1 - rad_eff2) / NE_layers)
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
            bound_val = pow(pow(x[0]/nucRad1,2) + pow(x[1]/nucRad1,2) + 
                        pow((x[2]-nucRad2-z0)/nucRad2,2),0.5)
            cutoffFrac = 0.99#1-0.95*thickness_thresh/rad_eff1
            return bound_val > cutoffFrac and on_boundary and np.sqrt(x[0]**2 + x[1]**2) < 0.8*npRad
            # return False
    class RoofInt(SubDomain): # for integrating over top
        def inside(self, x, on_boundary):
            bound_val = pow(pow(x[0]/nucRad1,2) + pow(x[1]/nucRad1,2) + 
                        pow((x[2]-nucRad2-z0)/nucRad2,2),0.5)
            cutoffFrac = 0.99#1-0.95*thickness_thresh/rad_eff1
            return bound_val > cutoffFrac and on_boundary and np.sqrt(x[0]**2 + x[1]**2) < 0.5*nucRad1 and x[2] > (z0 + nucRad2)
            # return False
    class SymmAxis1(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] < 0.001 and on_boundary
    class SymmAxis2(SubDomain):
        def inside(self, x, on_boundary):
            # return np.arctan(x[1]/x[0]) > 0.999*np.pi/4 and on_boundary
            return x[0] < 0.001 and on_boundary
    class SymmAxis3(SubDomain):
        def inside(self, x, on_boundary):
            # return np.arctan(x[1]/x[0]) > 0.999*np.pi/4 and on_boundary
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
    normal_expr = Expression(("(x[0]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b-z0,2)/pow(b,4))", 
                            "(x[1]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b-z0,2)/pow(b,4))",
                            "((x[2]-b-z0)/pow(b,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b-z0,2)/pow(b,4))"),
                                a=nucRad1, b=nucRad2, z0=z0, degree=1)
    normals = project(normal_expr, V_vector)

    # inner_normal_expr = Expression(("(x[0]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b-z0,2)/pow(b,4))", 
    #                         "(x[1]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b-z0,2)/pow(b,4))",
    #                         "((x[2]-b-z0)/pow(b,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b-z0,2)/pow(b,4))"),
    #                             a=nucRad1-rthickness, b=nucRad2-zthickness, z0=z0+zthickness, degree=1)
    # inner_normals = project(inner_normal_expr, V_vector)
    # V_vector_dof = vertex_to_dof_map(V_vector)
    # nvec = normals.vector()[:]
    # for f in facets(mesh):
    #     vidx = f.entities(0)
    #     if mf_surf[f] == 12:
    #         for v in vidx:
    #             dofcur = V_vector_dof[3*v:3*(v+1)]
    #             nvec[dofcur] = inner_normals.vector()[dofcur]
    # normals.vector().set_local(nvec)
    # normals.vector().apply("insert")
    inner_normals = normals # the same

    x = SpatialCoordinate(mesh)
    # results_folder = "/root/scratch/nuc_indent_1e6J_eighth"
    results_folder = args["outdir"]
    File(f"{results_folder}/test_shell.pvd") << mesh

    # Define mixed function space for displacements over each region
    el1 = VectorElement("P", mesh.ufl_cell(), 2) # u function space
    el2 = FiniteElement("P", mesh.ufl_cell(), 1) # p function space
    # el3 = FiniteElement("P", mesh_ne_outer.ufl_cell(), 1)
    mixed_element = MixedElement([el1, el2]) # mixed function space
    Vmixed = FunctionSpace(mesh, mixed_element)
    V1 = Vmixed.sub(0)
    # V_displ = VectorFunctionSpace(mesh, "P", 2)
    # V_pressure = FunctionSpace(mesh, "P", 1)
    V_psi = FunctionSpace(mesh_ne_outer, "P", 1) # psi_c function space for contact mechanics
    # Vmixed = MixedFunctionSpace(V_displ, V_pressure, V_psi)
    # V1 = Vmixed.sub_space(0)

    # Define Dirichlet boundary conditions
    V = FunctionSpace(mesh, "P", 1)
    # zDisplInitExpr = Expression("0.0", degree=1)
    # zDisplInit = interpolate(zDisplInitExpr, V)
    zDisplOuterExpr = Expression("-z0-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, r1=nucRad1, r2=nucRad2, z0=z0)
    zDisplOuter = interpolate(zDisplOuterExpr, V)
    zDisplFloorExpr = Expression("-z0-hNP-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, hNP=hNP, r1=nucRad1, r2=nucRad2, z0=z0)
    zDisplFloor = interpolate(zDisplFloorExpr, V)
    # zDisplInner = Expression("z1-(r2-(r2-z1)*sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, 
    #                         z1=zthickness, r1=nucRad1-rthickness, r2=nucRad2)
    # zDisplUpper = Expression("zMax-r2*(1+sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, zMax=zRoof, r1=nucRad1, r2=nucRad2)
    # zDisplUpperInner = Expression("zMax-(r2+(r2-z1)*sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, 
    #                             zMax=zRoof-zthickness, r1=nucRad1-rthickness, r2=nucRad2, z1=zthickness)
    uFixedExpr = Expression("0.0", degree=1)
    uxFixed = interpolate(uFixedExpr, V)
    uyFixed = interpolate(uFixedExpr, V)
    bc_nanopillar_z = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
    bc_nanopillar_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 1)
    bc_nanopillar_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 1)
    bc_side_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 8)
    bc_side_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 8)
    bc_floor = DirichletBC(V1.sub(2), zDisplFloor, mf_dirichlet, 7)
    # bc_nanopillar_inner = DirichletBC(V1.sub(2), zDisplInner, mf_dirichlet, 5)
    # bc_upper = DirichletBC(V1.sub(2), zDisplUpper, mf_dirichlet, 4)
    # bc_upper_inner = DirichletBC(V1.sub(2), zDisplUpperInner, mf_dirichlet, 6)
    # bc_fixed2 = DirichletBC(V1.sub(1), zDispl, fixed_bound2)
    bc_symm1_x = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 2)
    bc_symm1_y = DirichletBC(V1.sub(1), Constant(0.0), mf_dirichlet, 2)
    bc_symm2_x = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 3)
    bc_symm2_y = DirichletBC(V1.sub(1), Constant(0.0), mf_dirichlet, 3)
    # bc_symmEdge1 = DirichletBC(V1.sub(0), Constant(0.0), mf_edge, 1)
    # bc_symmEdge2 = DirichletBC(V1.sub(1), Constant(0.0), mf_edge, 1)
    bc_symm3_z = DirichletBC(V1.sub(2), Constant(0.0), mf_dirichlet2, 8)
    stick = False
    if stick:
        bcs = [bc_nanopillar_z, bc_nanopillar_x, bc_nanopillar_y, 
            bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
    else:
        # bcs = [bc_nanopillar_z, bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
        bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
    # bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z]

    # Define functions
    (du, dp) = TrialFunctions(Vmixed)            # Incremental displacement and dp
    (v, q)  = TestFunctions(Vmixed)             # Test function
    fmixed  = Function(Vmixed)                 # Displacement from previous iteration
    (u, p) = split(fmixed)
    psi_c = Function(V_psi)
    qpsi_c = TestFunction(V_psi)
    # dpsi_c = TrialFunction(Vmixed)

    domain_id = MeshFunction("size_t", mesh, 2, 0)
    for f in facets(mesh):
        if mf_dirichlet[f] == 2:
            domain_id[f] = 2
        elif mf_dirichlet[f] == 3:
            domain_id[f] = 3
        # elif f.midpoint().z() > 1.2*nucRad2 and mf_surf[f] == 10:
        #     domain_id[f] = 1
        # elif f.midpoint().z() > 1.2*nucRad2 and mf_surf[f] == 12:
        #     domain_id[f] = 4
        elif mf_dirichlet[f] == 1:
            domain_id[f] = 11
        elif mf_surf[f] == 10:
            domain_id[f] = 1
        elif mf_surf[f] == 12:
            domain_id[f] = 4
        # if (np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2) > 0.4 and 
        #     np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2) < 0.5):
        #         domain_id[f] = 1
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
    # E1, E2 = Constant(5000.0), Constant(0.0)

    # Stored strain energy density (incompressible Mooney Rivlin model)
    # psi = (mu/2)*(I1 - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
    psi = E1scale*E1*(I1-3) + E2scale*E2*(I2-3) - p*(J-1)
    psi2 = 0.1*psi
    # psi = E1*(I1-3) + 4e4*(J-1)**2

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
    innerBulkMod = 100.0 * args["softFactor"]
    pRamp = [pMin]
    p0 = Constant(0e4)
    dpress = 50.0# / soft_factor

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

    # u.set_allow_extrapolation(True)

    zIndentMax = args["nanopillar_height"]

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
    # Pmag = Expression("(1+exp(-x[2]/(0.2*nucRad2)))", degree=1, nucRad2=nucRad2)
    # Pinner = Expression(("0.0", "pressure"), degree=1, pressure=450)
    outer_area_factor = J * dot(normals, inv(F))
    inner_area_factor = J * dot(inner_normals, inv(F))
    T = forceConst*Tdir*sqrt(inner(outer_area_factor,outer_area_factor))
    Press_in = (p0 + pressConst)*inner_area_factor
    Press_out = p0 * outer_area_factor
    topContactForce = Function(V_vector)
    topContactForce_inner = Function(V_vector)

    # add steric forces at bottom
    V_vector_P2 = VectorFunctionSpace(mesh, "P", 2)
    Tsteric = Function(V_vector_P2)
    Tsteric_vec = Tsteric.vector()[:]
    stericCoords = V_vector_P2.tabulate_dof_coordinates()
    dSteric = 0.2
    stericMag = Constant(1000.0)
    stericCutoff = 0.01
    # npContactForce = (stericMag*0.5*(1-ufl.tanh((x[2]+u[2]-stericCutoff)/stericCutoff))*
    #                   sqrt(inner(outer_area_factor,outer_area_factor))*(-n_g)) # always repulsive
    nanopillar_logic = Constant(0.0)
    for i in range(len(xNP)):
        # nanopillar_logic += exp(-((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)/(2*npRad**2))
        nanopillar_logic += exp(-((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)**2/(npRad**4))
        # nanopillar_logic += (1-ufl.tanh((sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)-(npRad-stericCutoff))/(stericCutoff)))/2
    # nanopillar_logic = ((1-ufl.tanh((sqrt((x[0]+u[0]-xNP[0])**2+(x[1]+u[1]-yNP[0])**2)-npRad)/0.01))/2 +
    #                     (1-ufl.tanh((sqrt((x[0]+u[0]-xNP[1])**2+(x[1]+u[1]-yNP[1])**2)-npRad)/0.01))/2 +
    #                     (1-ufl.tanh((sqrt((x[0]+u[0]-xNP[2])**2+(x[1]+u[1]-yNP[2])**2)-npRad)/0.01))/2)
    npContactForce = nanopillar_logic*(stericMag*ufl.exp(-(x[2]+u[2]-dSteric)/stericCutoff)*
                      sqrt(inner(outer_area_factor,outer_area_factor))*(-n_g))
    
    # numFactor = 1e-6
    # for i in range(len(xNP)):
    #     if i==0:
    #         npSideForce_x = (((x[0]+u[0]-xNP[i])/(sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)+numFactor))
    #                         *ufl.exp(-(sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)-npRad-stericCutoff)/stericCutoff))
    #         npSideForce_y = (((x[1]+u[1]-yNP[i])/(sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)+numFactor))
    #                         *ufl.exp(-(sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)-npRad-stericCutoff)/stericCutoff))
    #     else:
    #         npSideForce_x += (((x[0]+u[0]-xNP[i])/(sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)+numFactor))
    #                         *ufl.exp(-(sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)-npRad-stericCutoff)/stericCutoff))
    #         npSideForce_y += (((x[1]+u[1]-yNP[i])/(sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)+numFactor))
    #                         *ufl.exp(-(sqrt((x[0]+u[0]-xNP[i])**2+(x[1]+u[1]-yNP[i])**2)-npRad-stericCutoff)/stericCutoff))
    # zLogic = (1-ufl.tanh((x[2]+u[2])/stericCutoff))/2
    # npSideForce_x *= zLogic*stericMag*sqrt(inner(outer_area_factor,outer_area_factor))
    # npSideForce_y *= zLogic*stericMag*sqrt(inner(outer_area_factor,outer_area_factor))
    
    # zHeaviside = (1+ufl.sign(x[2]+u[2]-stericCutoff))/2
    # npContactForce = (stericMag*(zHeaviside*(1/(x[2]+u[2]))**4 + (1/stericCutoff**4)*(1-zHeaviside))*
    #                   sqrt(inner(outer_area_factor,outer_area_factor))*(-n_g))

    # springHeaviside = (1-ufl.sign(x[2]+u[2]-dSteric))/2
    # npContactForce = (stericMag*(dSteric-(x[2]+u[2]))*springHeaviside*
    #                   sqrt(inner(outer_area_factor,outer_area_factor))*(-n_g))

    # Convert potential energy to first Piola-Kirchoff stress tensor
    dx = Measure("dx", domain=mesh, subdomain_data=mf_vol)
    Ttensor = diff(psi, F)
    Ttensor_in = diff(psi2, F)
    # alpha = Constant(0.1)
    # psi_c_mapped = sub_to_parent(psi_c, mesh)
    # psi_c_prev_mapped = sub_to_parent(psi_c_prev, mesh)
    Fvar = (inner(grad(v), Ttensor)*dx(1) - inner(v, stericMag*Tsteric)*ds(1) + inner(grad(v), Ttensor_in)*dx(2) -
            # inner(v[0], npSideForce_x)*ds(1) - inner(v[1], npSideForce_y)*ds(1) -
            inner(v, T+Press_out)*ds(1) - inner(v, topContactForce)*ds(1) - inner(v, npContactForce)*ds(1) -
            inner(v, topContactForce_inner+Press_in)*ds(4) - 
            inner(v, Press_in)*ds(12) - inner(v, Press_out)*ds(10) + 
            inner(q, args["bulk_mod"]*(J-1) + p) * dx(1) + inner(q, 0.1*args["bulk_mod"]*(J-1) + p) * dx(2))
    #         -inner(psi_c_mapped - psi_c_prev_mapped, dot(v, n_g)) * ds(11))
    # u_mapped = interpolate(fmixed.sub(0), VectorFunctionSpace(V_psi.mesh(), "P", 1))
    # outer_vertex_map = mesh_ne_outer.topology().mapping()[mesh.id()].vertex_map()
    # outer_facet_map = []
    # psi_domain_id = MeshFunction("size_t", mesh_ne_outer, 2, 0)
    # for c in cells(mesh_ne_outer):
    #     for f_global in facets(mesh):
    #         if mf_surf[f_global] == 10:
    #             if np.sum((c.midpoint()[:]-f_global.midpoint()[:])**2) < 1e-6:
    #                 outer_facet_map.append(f_global.index())
    #                 psi_domain_id[c] = domain_id[f_global]
    #                 break
    
    # dx_psi = Measure("dx", mesh_ne_outer, subdomain_data=psi_domain_id)
    # Fvar2 = (inner(dot(u_mapped, n_g), qpsi_c) * dx_psi
    #         + inner(exp(psi_c), qpsi_c) * dx_psi - inner(zDisplOuter, qpsi_c) * dx_psi)

            # - inner(1e6*(u-u_prev), v) * ds(11)) # positive for "extra slip"

    # save initial stresses
    surf_stress = project(dot(normals, Ttensor)/sqrt(inner(outer_area_factor,outer_area_factor)), V_vector)
    surf_stress_file.write(surf_stress, idx)
    psi_fcn = project(psi, V_scalar)
    psi_file.write(psi_fcn, idx)
    # + inner(q, u[0]-u[1])*ds(3) + inner(q, u[1])*ds(2)
    # Compute Jacobian of F
    dw = TrialFunction(Vmixed)
    Jvar = derivative(Fvar, fmixed)#, dw)
    # Jvar2 = derivative(Fvar2, psi_c)
    # Define problem and solver with custom settings
    custom_solver = False
    if custom_solver:
        problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
    else:
        solver = init_solver(Fvar, fmixed, bcs, Jvar)

    # solver2 = init_solver(Fvar2, psi_c, [], Jvar2)
    # load cell without nucleus
    loaded = mesh_tools.load_mesh(args["mesh_folder"] / "spreadCell_mesh.h5")
    # loaded = mesh_tools.load_mesh("/root/shared/gitrepos/smart-mechanotransduction"
    #                               "/mesh-files/mesh_largeNPquarter/spreadCell_mesh.h5")
    cell_mesh = loaded.mesh
    cell_mf2 = loaded.mf_facet
    cell_mf3 = loaded.mf_cell
    mesh_pm = create_meshview(cell_mf2, 10)
    mesh_cyto = create_meshview(cell_mf3, 1)
    if nuc_only:
        zRoofPM = 0.0
    else:
        zRoofPM = max(mesh_pm.coordinates()[:,2])
    dSteric = 0.2
    zShift = hNP + dSteric

    V1lin = VectorFunctionSpace(mesh, "P", 1)
    linFcn = interpolate(fmixed.sub(0), V1lin)
    uLagrange, volLagrange = lagrangeMap(mf3, 2, mf2, [10,12], 
                                        [linFcn,linFcn], 1/4, None)
    vol_inner = [volLagrange]

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
        # if kRamp[-1] > 100.0:
            # E_curExpr = Expression("5000.0*(1-0.5*(1-exp(-(k-k0)/k1))*exp(-x[2]/2.0))", 
            #                         degree=1, k=kRamp[-1], k0=100.0, k1=200.0)
            # E1.assign(project(E_curExpr, V))
            # Vcoords = V.tabulate_dof_coordinates()
            # E1vec = E1.vector()[:]
            # for i in range(len(Vcoords)):
            #     if Vcoords[i][2] > 7.5:
            #         E1vec[i] *= 0.9
            # E1.vector().set_local(E1vec)
            # E1.vector().apply("insert")
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
                pressConst.assign(curPress)
                reset = False
                try:
                    if custom_solver:
                        solver.solve(None, problem.fmixed.vector().vec())
                        if solver.getConvergedReason() <= 0:
                            reset = True
                        else:
                            set_step = True
                    else:
                        # uz_loops = 0
                        # while True:
                        #     u_previt = fmixed.sub(0).copy(deepcopy=True)
                        #     psi_previt = psi_c.copy(deepcopy=True)
                        #     solver.solve()
                        #     delta_u = sqrt(assemble(pow(u_previt-fmixed.sub(0),2)*dx))
                        #     u_mapped.assign(interpolate(fmixed.sub(0), VectorFunctionSpace(V_psi.mesh(), "P", 1)))
                        #     solver2.solve()
                        #     delta_psi = sqrt(assemble(pow(psi_previt-psi_c,2)*dx_psi(11)))
                        #     psi_c_mapped.assign(sub_to_parent(psi_c, mesh))
                        #     psi_c_prev_mapped.assign(sub_to_parent(psi_c_prev, mesh))
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

            if it >= 10:
                logger.info(f"Done computing idx = {idx} after maximum ({it}) iterations, def = {uEval(0,0,2*nucRad2)[2]}")
                break
            fmixed_prev = fmixed.vector()[:].copy()
            coords = mf_dirichlet.mesh().coordinates()

            # continueSmooth = True
            # while continueSmooth:
            # uref = fmixed.sub(0).copy(deepcopy=True)
            # pref = fmixed.sub(1).copy(deepcopy=True)
            # diffStep = 0.01
            # Fdiffuse = inner(grad(u), grad(v))*dx(1) + (J*inner(u-uref,v)/diffStep)*dx(1) + (p-pref)*q*dx(1)
            # Jdiffuse = derivative(Fdiffuse, fmixed)
            # problem_diffusion = NonlinearVariationalProblem(Fdiffuse, fmixed, bcs, J=Jdiffuse)
            # solver_diffusion = NonlinearVariationalSolver(problem_diffusion)
            # prm = solver_diffusion.parameters
            # prm["newton_solver"]["absolute_tolerance"] = 1E-6
            # prm["newton_solver"]["relative_tolerance"] = 1E-6
            # prm["newton_solver"]["maximum_iterations"] = 100
            # prm["newton_solver"]["linear_solver"] = 'mumps'
            # solver_diffusion.solve()
            ulin.assign(project(fmixed.sub(0), V_vector))
            testJacobian = project(det(Identity(3) + grad(ulin)), V_scalar)
            print(f"Current min Jacobian is {min(testJacobian.vector())}")

            reinit, keepSwimming, mf_dirichlet, domain_id, uxFixed, uyFixed = update_bcs(
                        fmixed, mf_dirichlet, domain_id, mf_surf, nanopillar_specs, uxFixed, uyFixed)
            if not pumpedUp:
                keepSwimming = False
            def stericCalc(x):
                # if x > dSteric:
                #     return 0.0
                # else:
                #     return (dSteric-x)
                # return 0.5*(1-np.tanh((x-stericCutoff)/stericCutoff))
                return 0.0#np.exp(-(x-dSteric)/stericCutoff)
            for i in range(0,len(stericCoords),3):
                xCur = stericCoords[i,:]
                bound_val = np.sqrt((xCur[0]/nucRad1)**2 + (xCur[1]/nucRad1)**2 + ((xCur[2]-nucRad2-z0)/nucRad2)**2)
                if bound_val > 0.99:   
                    uCur = uEval(xCur)
                    xDef = xCur + uCur
                    if xDef[2] <= 10.0:
                        all_dist = np.sqrt((xNP-xDef[0])**2 + (yNP-xDef[1])**2)
                        if np.any(all_dist <= npRad) or (len(xNP)==0):
                            Tsteric_vec[i:i+3] = [0.0, 0.0, stericCalc(xDef[2])]
                            # if xDef[2] < 0.0:
                            #     print("uh oh")
                            # if xDef[2] > stericCutoff:
                            #     Tsteric_vec[i:i+3] = [0.0, 0.0, 1/xDef[2]]
                            # else:
                            #     Tsteric_vec[i:i+3] = [0.0, 0.0, 1/stericCutoff]
                        else:
                            min_idx = np.argmin(all_dist)
                            if xDef[2] > 0.0:
                                side_dist = np.sqrt((all_dist[min_idx]-npRad)**2 + xDef[2]**2)
                                rScale = (all_dist[min_idx]-npRad)/side_dist
                                xDir = -rScale*(xNP[min_idx] - xDef[0])/all_dist[min_idx]
                                yDir = -rScale*(yNP[min_idx] - xDef[1])/all_dist[min_idx]
                                zDir = xDef[2]/side_dist
                            else:
                                side_dist = all_dist[min_idx] - npRad
                                # if side_dist <= stericCutoff:
                                #     # logger.warning("Shape is folding inwards!")
                                #     side_dist = stericCutoff # lower cutoff
                                xDir = -(xNP[min_idx] - xDef[0])/all_dist[min_idx]
                                yDir = -(yNP[min_idx] - xDef[1])/all_dist[min_idx]
                                zDir = 0.0
                            bottom_dist = xDef[2] + hNP
                            if side_dist < (bottom_dist+dSteric):
                                force_mag = stericCalc(side_dist)#1/side_dist #np.exp(-side_dist/dSteric)
                                Tsteric_vec[i:i+3] = [force_mag*xDir, force_mag*yDir, force_mag*zDir]
                            else:
                                Tsteric_vec[i:i+3] = [0.0, 0.0, stericCalc(bottom_dist+dSteric)]#1/(bottom_dist+dSteric)]#np.exp(-(bottom_dist+dSteric)/dSteric)]
            Tsteric.vector().set_local(Tsteric_vec)
            Tsteric.vector().apply("insert")
            Tsteric_conv = project(Tsteric*sqrt(inner(outer_area_factor,outer_area_factor)), V_vector_P2)
            Tsteric.assign(Tsteric_conv)

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

                pressIts.append(pressIts[0] - innerBulkMod*(volLagrange-vol_inner[-1])/volLagrange)
                pRamp[-1] = pressIts[-1]
                if pressIts[-2] > 0.0:
                    if np.abs((pressIts[-1]-pressIts[-2])/pressIts[-2]) > 0.01:
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
                    zClampCur = (assemble(uEval[2]*sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1))/
                                 assemble(sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1)))#uEval(0,npRad,z0)[2]
                    bc_symm3_z = DirichletBC(V1.sub(2), Constant(zClampCur), mf_dirichlet2, 8)
                # bc_nanopillar_inner = DirichletBC(V1.sub(2), zDisplInner, mf_dirichlet, 5)
                # bc_upper = DirichletBC(V1.sub(2), zDisplUpper, mf_dirichlet, 4)
                # bc_upper_inner = DirichletBC(V1.sub(2), zDisplUpperInner, mf_dirichlet, 6)
                if stick:
                    bcs = [bc_nanopillar_x, bc_nanopillar_y, bc_nanopillar_z, 
                        bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
                else:
                    # bcs = [bc_nanopillar_z, bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
                    bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
                
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
            if not inCell:
                intersect_logic = False #check_intersection(fmixed, mesh_ne_outer, mesh_cyto, zShift)
                if not intersect_logic:
                    inCell = True
                    eighth_coords = ne_mesh_eighth.coordinates()[:]
                    Veighth = VectorFunctionSpace(ne_mesh_eighth, "P", 1)
                    u_eighth = Function(Veighth)
                    u_eighth_vec = u_eighth.vector()[:]
                    u_eighth_dof = vertex_to_dof_map(Veighth)
                    u_eighth_inner = Function(Veighth)
                    u_eighth_inner_vec = u_eighth_inner.vector()[:]
                    for i in range(len(eighth_coords)):
                        xcur = eighth_coords[i]
                        ucur = uEval(xcur)
                        rcur = np.sqrt(xcur[0]**2 + xcur[1]**2 + (xcur[2]-nucRad2)**2)
                        if np.abs(rcur - nucRad1) > 0.02*nucRad1:
                            logger.info(f"rcur is {rcur}!")
                            raise ValueError("Not at the right point")
                        if nucRad1 != nucRad2 or rthickness != zthickness:
                            raise ValueError("inner extrapolation does not work for this geo")
                        xcur_inner = xcur * (nucRad2-rthickness)/nucRad2
                        xcur_inner[2] = nucRad2 + (xcur[2]-nucRad2) * (nucRad2-rthickness)/nucRad2
                        uShift_inner = xcur_inner - xcur
                        ucur_inner = uEval(xcur_inner)
                        if np.isclose(eighth_coords[i,1], 0.0):
                            ucur[1] = 0.0
                            ucur_inner[1] = 0.0
                        elif np.isclose(eighth_coords[i,0], 0.0):
                            ucur[0] = 0.0
                            ucur_inner[0] = 0.0
                        elif np.isclose(np.arctan2(eighth_coords[i,1],eighth_coords[i,0]), np.pi/4):
                            uxy = (ucur[0]+ucur[1])/2
                            ucur[0] = uxy
                            ucur[1] = uxy
                            uxy_inner = (ucur_inner[0]+ucur_inner[1])/2
                            ucur_inner[0] = uxy_inner
                            ucur_inner[1] = uxy_inner
                        ucur[2] += zShift
                        ucur_inner[2] += zShift  
                        eighth_coords[i] += ucur
                        u_eighth_vec[u_eighth_dof[3*i:3*(i+1)]] = ucur
                        u_eighth_inner_vec[u_eighth_dof[3*i:3*(i+1)]] = ucur_inner - uShift_inner
                    u_eighth.vector().set_local(u_eighth_vec)
                    u_eighth.vector().apply("insert")
                    u_eighth.set_allow_extrapolation(True)
                    u_eighth_inner.vector().set_local(u_eighth_inner_vec)
                    u_eighth_inner.vector().apply("insert")
                    u_eighth_inner.set_allow_extrapolation(True)

                    (mesh_full, mf2_full, mf3_full, substrate_markers, curv_markers) \
                        = mesh_gen.assemble_ne_pm(ne_mesh=ne_mesh_eighth, pm_mesh=mesh_pm, 
                                                  hEdge=0.5, hNP = 0.3,
                                                  nanopillars=nanopillar_specs, sym_fraction=1/4,
                                                  use_tmp=True)
                    # now save this mesh in the designated folder
                    mesh_folder = args["mesh_folder"] / "deformed"
                    mesh_folder.mkdir(exist_ok=True, parents=True)
                    mesh_file = mesh_folder / "spreadCell_mesh.h5"
                    mesh_tools.write_mesh(mesh_full, mf2_full, mf3_full, mesh_file, [substrate_markers])
                    File(str(mesh_folder / "facets.pvd")) << mf2_full
                    File(str(mesh_folder / "cells.pvd")) << mf3_full
                    # save curvatures for reference
                    curv_file_name = mesh_folder / "curvatures.xdmf"
                    with XDMFFile(str(curv_file_name)) as curv_file:
                        curv_file.write(curv_markers)
                    # get submeshes and subspaces
                    mesh_ne_full = create_meshview(mf2_full, 12)
                    V1lin_full = VectorFunctionSpace(mesh_ne_full, "P", 1)
                    u_start = Function(V1lin_full)
                    u_start_vec = u_start.vector()[:]
                    u_start_inner = Function(V1lin_full)
                    u_start_inner_vec = u_start_inner.vector()[:]
                    start_coords = V1lin_full.tabulate_dof_coordinates()
                    for i in range(0,len(start_coords),3):
                        all_dist = np.sqrt((start_coords[i,0]-eighth_coords[:,0])**2 +
                                        (start_coords[i,1]-eighth_coords[:,1])**2 +
                                        (start_coords[i,2]-eighth_coords[:,2])**2)
                        closest_idx = np.argmin(all_dist)
                        if all_dist[closest_idx] < 1e-6:
                            idx_list = u_eighth_dof[3*closest_idx:3*(closest_idx+1)]
                            u_start_vec[i:i+3] = u_eighth_vec[idx_list]
                            u_start_inner_vec[i:i+3] = u_eighth_inner_vec[idx_list]
                        else:
                            interp_done = False
                            for c in cells(ne_mesh_eighth):
                                cur_dist = c.distance(Point(start_coords[i]))
                                if cur_dist < 1e-6: # then use this element
                                    # determine if on edge or in interior
                                    for e in edges(c):
                                        vidx = e.entities(0)
                                        test_vec = start_coords[i] - eighth_coords[vidx[0]]
                                        ref_vec = eighth_coords[vidx[1]] - eighth_coords[vidx[0]]
                                        orth_line = np.cross(test_vec, ref_vec) / np.linalg.norm(ref_vec)
                                        if np.linalg.norm(orth_line) < 1e-3: # then on or close to edge
                                            scale1 = np.dot(test_vec, ref_vec) / np.linalg.norm(ref_vec)
                                            scale1 /= e.length()
                                            scale0 = 1 - scale1
                                            idx_list_0 = u_eighth_dof[3*vidx[0]:3*(vidx[0]+1)]
                                            idx_list_1 = u_eighth_dof[3*vidx[1]:3*(vidx[1]+1)]
                                            u_start_vec[i:i+3] = scale0*u_eighth_vec[idx_list_0] + scale1*u_eighth_vec[idx_list_1]
                                            u_start_inner_vec[i:i+3] = (scale0*u_eighth_inner_vec[idx_list_0] + 
                                                                        scale1*u_eighth_inner_vec[idx_list_1])
                                            interp_done = True
                                            break
                                    if interp_done:
                                        break
                                    else:
                                        logger.warning("Still need to fix this approximation!")
                                        vidx = c.entities(0)
                                        idx_list_0 = u_eighth_dof[3*vidx[0]:3*(vidx[0]+1)]
                                        idx_list_1 = u_eighth_dof[3*vidx[1]:3*(vidx[1]+1)]
                                        idx_list_2 = u_eighth_dof[3*vidx[2]:3*(vidx[2]+1)]
                                        u_start_vec[i:i+3] = (u_eighth_vec[idx_list_0] + 
                                                            u_eighth_vec[idx_list_1] + 
                                                            u_eighth_vec[idx_list_2])/3
                                        u_start_inner_vec[i:i+3] = (u_eighth_inner_vec[idx_list_0] + 
                                                                    u_eighth_inner_vec[idx_list_1] + 
                                                                    u_eighth_inner_vec[idx_list_2])/3
                                    break

                    u_start.vector().set_local(u_start_vec)
                    u_start.vector().apply("insert")
                    u_start_inner.vector().set_local(u_start_inner_vec)
                    u_start_inner.vector().apply("insert")
                    linFcn_full = Function(V1lin_full)
                    aLagrangeFull = Function(V1lin_full)
                    aLagrangeInner = Function(V1lin_full)
                    uLagrangeInner = Function(V1lin_full)
                    linFcnCoords = V1lin_full.tabulate_dof_coordinates()
                    linFcnVec = linFcn_full.vector()[:]
                    for i in range(0,len(linFcnCoords),3):
                        xcur = linFcnCoords[i] - u_start_vec[i:i+3]
                        ucur = uEval(xcur)
                        linFcnVec[i:i+3] = ucur - u_start_vec[i:i+3] # deformation w.r.t. starting geo in cell
                        linFcnVec[i+2] += zShift
                    linFcn_full.vector().set_local(linFcnVec)
                    linFcn_full.vector().apply("insert")

                    manual_map = False
                    if manual_map:
                        # create scale values throughout cytosol
                        scale_vec = np.zeros(mesh_full.num_vertices())
                        compute_logic = np.zeros_like(scale_vec)
                        closest_ne_indices = np.zeros_like(scale_vec)
                        all_cyto_indices = []
                        ne_coords = mesh_ne_full.coordinates()
                        ne_map = mesh_ne_full.topology().mapping()[mesh_full.id()].vertex_map()
                        pm_coords = create_meshview(mf2_full, 10).coordinates()
                        for c in cells(mesh_full):
                            if mf3_full[c] == 1: # then cytosol
                                for f in facets(c):
                                    for v in vertices(f):
                                        if not compute_logic[v.index()]:
                                            ne_dist = np.sqrt((v.x(0)-ne_coords[:,0])**2 +
                                                            (v.x(1)-ne_coords[:,1])**2 +
                                                            (v.x(2)-ne_coords[:,2])**2)
                                            ne_idx = np.argmin(ne_dist)
                                            closest_ne_indices[v.index()] = int(ne_map[ne_idx])
                                            all_cyto_indices.append(v.index())
                                            if mf2_full[f] == 10: # then PM
                                                scale_vec[v.index()] = 0.0
                                            elif mf2_full[f] == 12: # then NE
                                                scale_vec[v.index()] = 1.0
                                            else:
                                                pm_dist = np.sqrt((v.x(0)-pm_coords[:,0])**2 +
                                                                (v.x(1)-pm_coords[:,1])**2 +
                                                                (v.x(2)-pm_coords[:,2])**2)
                                                pm_dist = np.min(pm_dist)
                                                scale_vec[v.index()] = pm_dist / (pm_dist + ne_dist[ne_idx])
                                            compute_logic[v.index()] = 1

                        uLagrangeFull = sub_to_parent(linFcn_full, mesh_full)
                        ufull_vec = uLagrangeFull.vector()[:]
                        ufull_dof = vertex_to_dof_map(uLagrangeFull.function_space())
                        all_cyto_indices = np.array(all_cyto_indices)
                        closest_ne_indices = closest_ne_indices.astype(int)
                        for offset in [0,1,2]:
                            curidx = closest_ne_indices[all_cyto_indices]
                            ufull_vec[ufull_dof[3*all_cyto_indices+offset]] = scale_vec[all_cyto_indices] * ufull_vec[ufull_dof[3*curidx+offset]]
                        
                        nuc_mesh = create_meshview(mf3_full, 2)
                        nuc_map = np.array(nuc_mesh.topology().mapping()[mesh_full.id()].vertex_map())
                        mf3_nuc = MeshFunction("size_t", nuc_mesh, 3, 2)
                        mf2_nuc = MeshFunction("size_t", nuc_mesh, 2)
                        class allBound(SubDomain):
                            def inside(self, x, on_boundary):
                                return on_boundary
                        markBound = allBound()
                        markBound.mark(mf2_nuc, 12)
                        for f in facets(nuc_mesh):
                            # theta_cur = np.arctan2(f.midpoint().y(),f.midpoint().x())
                            if f.midpoint().y() == 0.0 or f.midpoint().x() == 0: # theta_cur > .2499*np.pi:
                                mf2_nuc[f] = 0

                        V1nuc = VectorFunctionSpace(nuc_mesh, "P", 1)
                        linFcn_nuc = interpolate(uLagrangeFull, V1nuc)

                        uLagrangeNuc, volNuc = lagrangeMap(mf3_nuc, 2, mf2_nuc, [12], 
                                                    [linFcn_nuc], 1/8, None, degree=1)
                        
                        nucdof = dof_to_vertex_map(V1nuc)
                        nuc_xindices = ufull_dof[3*nuc_map[np.floor(nucdof[::3]/3).astype(int)]]
                        nuc_yindices = nuc_xindices + 1
                        nuc_zindices = nuc_xindices + 2
                        ufull_vec[nuc_xindices] = uLagrangeNuc.vector()[::3]
                        ufull_vec[nuc_yindices] = uLagrangeNuc.vector()[1::3]
                        ufull_vec[nuc_zindices] = uLagrangeNuc.vector()[2::3]
                        uLagrangeFull.vector().set_local(ufull_vec)
                        uLagrangeFull.vector().apply("insert")
                    else:
                        closest_ne_indices = None
                        all_cyto_indices = None 
                        scale_vec = None 
                        uLagrangeNuc = None
                        uLagrangeFull, volFull = lagrangeMap(mf3_full, 2, mf2_full, [10,12], 
                                        [linFcn_full,linFcn_full], 1/4, None)

                    # vol_inner.append(volLagrange)
                    # pNew = pRamp[-1]            
                    uFull_file = XDMFFile(f"{results_folder}/u_np_fullcell.xdmf")
                    uFull_file.parameters["flush_output"] = True
                    aFull_file = XDMFFile(f"{results_folder}/a_np_fullcell.xdmf")
                    aFull_file.parameters["flush_output"] = True
                    aInner_file = XDMFFile(f"{results_folder}/a_np_inner.xdmf")
                    aInner_file.parameters["flush_output"] = True
                    uInner_file = XDMFFile(f"{results_folder}/u_np_inner.xdmf")
                    uInner_file.parameters["flush_output"] = True
                    uFull_file.write(uLagrangeFull, 0.0)
        
        # set next force values
        linFcn = interpolate(fmixed.sub(0), V1lin)
        uLagrange, volLagrange = lagrangeMap(mf3, 2, mf2, [10,12], 
                                    [linFcn,linFcn], 1/4, uLagrange)

        vol_inner.append(volLagrange)
        pNew = pRamp[-1] - innerBulkMod*(vol_inner[-1]-vol_inner[-2])/vol_inner[-1]
        
        if pRamp[-1] >= pMax and not pumpedUp:
            pumpedUp = True
            kRamp.append(kMin)
            pRamp.append(pMax)
            # mesh.coordinates()[:,2] += np.abs(np.min(uEval.vector()[::2]))
            # mesh_ref.coordinates()[:,2] += np.abs(np.min(uEval.vector()[::2]))
            # mesh_ref_eighth.coordinates()[:,2] += np.abs(np.min(uEval.vector()[::2]))
            # ne_mesh_eighth.coordinates()[:,2] += np.abs(np.min(uEval.vector()[::2]))
            # reinit, keepSwimming, mf_dirichlet, domain_id, uxFixed, uyFixed = update_bcs(
            #             fmixed, mf_dirichlet, domain_id, mf_surf, nanopillar_specs, uxFixed, uyFixed)
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
            # bc_nanopillar_inner = DirichletBC(V1.sub(2), zDisplInner, mf_dirichlet, 5)
            # bc_upper = DirichletBC(V1.sub(2), zDisplUpper, mf_dirichlet, 4)
            # bc_upper_inner = DirichletBC(V1.sub(2), zDisplUpperInner, mf_dirichlet, 6)
            if stick:
                bcs = [bc_nanopillar_x, bc_nanopillar_y, bc_nanopillar_z, 
                    bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
            else:
                # bcs = [bc_nanopillar_z, bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
                bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
            # bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z]
            if custom_solver:
                problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
            else:
                solver = init_solver(Fvar, fmixed, bcs, Jvar)
        elif pumpedUp:
            # zRoof = 2*nucRad2 + u(0,0,2*nucRad2)[2]
            if kRamp[-1] < kMax:
                kRamp.append(min([kRamp[-1]+kInc, kMax]))
                pRamp.append(pNew)
            # elif hNP < hNPMax:
            #     kRamp.append(kMax)
            #     pRamp.append(pMax)
            #     hNP = hNPMax
            #     nanopillar_specs["hNP"] = hNP
            #     reinit, keepSwimming, mf_dirichlet, domain_id = update_bcs(
            #         fmixed, mf_dirichlet, domain_id, mf_surf, nanopillar_specs)
            #     if reinit: # then solver needs to be updated
            #         zDisplFloorExpr = Expression("-z0-hNP-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", 
            #                                  degree=1, hNP=hNP, r1=nucRad1, r2=nucRad2, z0=z0)
            #         zDisplFloor = interpolate(zDisplFloorExpr, V)
            #         bc_nanopillar = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
            #         bc_floor = DirichletBC(V1.sub(2), zDisplFloor, mf_dirichlet, 7)
            #         bcs = [bc_nanopillar, bc_floor, bc_symm1_y, bc_symm2_x]
            #         if custom_solver:
            #             problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
            #         else:
            #             solver = init_solver(Fvar, fmixed, bcs, Jvar)
            else:
                stopLogic = True # then done with this simulation
        else:
            pRamp.append(min([pRamp[-1]+dpress, pMax]))
            kRamp.append(kMin)   
        u_file.write_checkpoint(fmixed.sub(0), "u_np_ellipsoid", idx, append=True)
        ulin_file.write(ulin, idx)
        # if min(testJacobian.vector()) < 0.0:
        #     raise ValueError("This is bad")

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

        # zNP.append((zNP[-1]+u(0,0,0)[2])/2)
        # kRepel += 0.01

        # compute stretch for current configuration
        a_vector_new = J * dot(normals, inv(F))
        a_vector_new = project(a_vector_new, V_vector)
        a_vector.assign(a_vector_new)
        a_file.write(a_vector, idx)

        # save stress and strain energy density for current configuration
        surf_stress.assign(project(dot(normals, Ttensor), V_vector))
        surf_stress_file.write(surf_stress, idx)
        psi_fcn.assign(project(psi, V_scalar))
        psi_file.write(psi_fcn, idx)

        if inCell:
            a_vector.set_allow_extrapolation(True)
            aFullVec = aLagrangeFull.vector()[:]
            aInnerVec = aLagrangeInner.vector()[:]
            uInnerVec = uLagrangeInner.vector()[:]
            for i in range(0,len(linFcnCoords),3):
                xcur = linFcnCoords[i] - u_start_vec[i:i+3]
                acur = a_vector(xcur)
                aFullVec[i:i+3] = acur
                xcur_inner = linFcnCoords[i] - u_start_inner_vec[i:i+3]
                # rcur = np.sqrt(xcur[0]**2 + xcur[1]**2 + (xcur[2]-nucRad2)**2)
                # if np.abs(rcur - nucRad1) > 0.01*nucRad1:
                #     raise ValueError("Not at the right point")
                # if nucRad1 != nucRad2 or rthickness != zthickness:
                #     raise ValueError("inner extrapolation does not work for this geo")
                # xcur_inner = xcur * (nucRad2-rthickness)/nucRad2
                # xcur_inner[2] = nucRad2 + (xcur[2]-nucRad2) * (nucRad2-rthickness)/nucRad2
                acur_inner = a_vector(xcur_inner)
                ucur_inner = uEval(xcur_inner)
                aInnerVec[i:i+3] = acur_inner
                uInnerVec[i:i+3] = ucur_inner - u_start_inner_vec[i:i+3]
                uInnerVec[i+2] += zShift
            aLagrangeFull.vector().set_local(aFullVec)
            aLagrangeFull.vector().apply("insert")
            aLagrangeInner.vector().set_local(aInnerVec)
            aLagrangeInner.vector().apply("insert")
            uLagrangeInner.vector().set_local(uInnerVec)
            uLagrangeInner.vector().apply("insert")
            aFull_file.write(aLagrangeFull, idx)
            aInner_file.write(aLagrangeInner, idx)
            uInner_file.write(uLagrangeInner, idx)

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
    
    if inCell:
        nuc_dict = {"idx": idx, "pRamp": pRamp, "kRamp": kRamp, "forceConst": forceConst, "kInc": kInc,
                    "pressConst": pressConst, "fmixed": fmixed, "solver": solver, "Fvar": Fvar,
                    "Jvar": Jvar, "bcs": bcs, "mf_dirichlet": mf_dirichlet, "mf_dirichlet2": mf_dirichlet2, "domain_id": domain_id, 
                    "mf_surf": mf_surf, "nanopillar_specs": nanopillar_specs, "uxFixed": uxFixed, "uyFixed": uyFixed,
                    "u_start": u_start, "u_start_inner": u_start_inner, "mf3_full": mf3_full, "mf2_full": mf2_full, "vol_inner": vol_inner,
                    "nucRad1": nucRad1, "nucRad2": nucRad2, "zDisplOuter": zDisplOuter, 
                    "dSteric": dSteric, "u_prev": u_prev, "a_vector": a_vector, "u_file": u_file, "a_file": a_file, 
                    "uFull_file": uFull_file, "aFull_file": aFull_file, "aInner_file": aInner_file, "uInner_file": uInner_file,
                    "results_folder": results_folder, "aLagrangeFull": aLagrangeFull, "aLagrangeInner": aLagrangeInner, 
                    "uLagrangeFull": uLagrangeFull, "uLagrangeInner": uLagrangeInner, "inner_SA": inner_SA, "outer_SA": outer_SA, "vol": vol, 
                    "volLagrange": volLagrange, "rthickness": rthickness, "zthickness": zthickness, 
                    "fmixed_prev": fmixed_prev, "mf3": mf3, "mf2": mf2, "uLagrange": uLagrange,
                    "closest_ne_indices": closest_ne_indices, "all_cyto_indices": all_cyto_indices, 
                    "scale_vec": scale_vec, "uLagrangeNuc": uLagrangeNuc, "innerBulkMod": innerBulkMod,
                    "ulin": ulin, "ulin_file": ulin_file, "zTops": zTops, "nuc_only": nuc_only,
                    "surf_stress": surf_stress, "surf_stress_file": surf_stress_file, "psi_fcn": psi_fcn, "psi_file": psi_file,
                    "E1scale": E1scale, "E2scale": E2scale, "psi_c_prev": psi_c_prev, "ds_roofInt": ds_roofInt}
    else:
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
                    "surf_stress": surf_stress, "surf_stress_file": surf_stress_file, "psi_fcn": psi_fcn, "psi_file": psi_file,
                    "E1scale": E1scale, "E2scale": E2scale, "psi_c_prev": psi_c_prev, "ds_roofInt": ds_roofInt}
    
    return nuc_dict

def init_solver(Fvar, fmixed, bcs, Jvar):
    problem = NonlinearVariationalProblem(Fvar, fmixed, bcs, J=Jvar)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-6
    prm["newton_solver"]["relative_tolerance"] = 1E-6
    prm["newton_solver"]["maximum_iterations"] = 100
    prm["newton_solver"]["linear_solver"] = 'mumps'
    # print(f'Solver params: {dict(solver.parameters)}')
    # print(f'Newton solver params: {dict(solver.parameters["newton_solver"])}')
    # print(f'Krylov solver params: {dict(solver.parameters["newton_solver"]["krylov_solver"])}')
    # print(f'LU solver params: {dict(solver.parameters["newton_solver"]["lu_solver"])}')
    # print(f'SNES solver params: {dict(solver.parameters["snes_solver"])}')
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
    E2_nominal = 5000.0


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

    # Dirichlet BCs
    # testBound = MeshFunction("size_t", mesh, 2, 0)
    # class allBound(SubDomain):
    #     def inside(self, x, on_boundary):
    #         return on_boundary
    # markBound = allBound()
    # markBound.mark(testBound, 1)
    # for f in facets(mesh):
    #     if mf_bc[f] == 0 and testBound[f] == 1:
    #         mf_bc[f] = 1
    ds_global = Measure("ds", domain=mesh, subdomain_data=mf_bc)
    # uBoundExpr = Expression(("0.0","0.0","1.0"),degree=1)
    # uBound = interpolate(uBoundExpr, VectorFunctionSpace(create_meshview(mf_dirichlet,1), "P", 1))
    # uBound = sub_to_sub(uBounds[0], mesh, mf3.mesh())
    bcs = []
    for i in range(len(uBounds)):
        if uBounds[i].function_space().mesh().id() != mesh.id():
            uBound = sub_to_parent(uBounds[i], mesh)
        else:
            uBound = uBounds[i]
        bcs.append(DirichletBC(V, uBound, mf_bc, bound_ids[i]))
    # bcs = []
    # for i in range(len(uBounds)):
    #     uBounds[i] = sub_to_parent(uBounds[i], mesh)
    #     bcs.append(DirichletBC(V, uBounds[i], mf2, bound_ids[i]))
    
    dx_global = Measure("dx", domain=mesh, subdomain_data=mf3)

    # Total potential energy
    # domain_markers = np.unique(mf3.array())
    # Pi_global = 0.0
    # for i in range(len(domain_markers)):
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
    psi_fcn = nuc_dict["psi_fcn"]
    psi_file = nuc_dict["psi_file"]
    E1scale = nuc_dict["E1scale"]
    E2scale = nuc_dict["E2scale"]
    psi_c_prev = nuc_dict["psi_c_prev"]
    ds_roofInt = nuc_dict["ds_roofInt"]
    stick = False

    if not nuc_dict["nuc_only"]:
        u_start = nuc_dict["u_start"]
        u_start_inner = nuc_dict["u_start_inner"]
        mf3_full = nuc_dict["mf3_full"]
        mf2_full = nuc_dict["mf2_full"]
        uFull_file = nuc_dict["uFull_file"]
        aFull_file = nuc_dict["aFull_file"]
        aInner_file = nuc_dict["aInner_file"]
        uInner_file = nuc_dict["uInner_file"]
        aLagrangeFull = nuc_dict["aLagrangeFull"]
        aLagrangeInner = nuc_dict["aLagrangeInner"]
        uLagrangeFull = nuc_dict["uLagrangeFull"]
        uLagrangeInner = nuc_dict["uLagrangeInner"]
        closest_ne_indices = nuc_dict["closest_ne_indices"]
        all_cyto_indices = nuc_dict["all_cyto_indices"]
        scale_vec = nuc_dict["scale_vec"]
        uLagrangeNuc = nuc_dict["uLagrangeNuc"]

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

    # V = zDisplOuter.function_space()
    # zDisplFloorExpr = Expression("-z0-hNP-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", 
    #                                     degree=1, hNP=hNP, r1=nucRad1, r2=nucRad2,z0=z0)
    # zDisplFloor = interpolate(zDisplFloorExpr, V)
    # bc_nanopillar_z = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
    # bc_nanopillar_x = DirichletBC(V1.sub(0), uxFixed, mf_dirichlet, 1)
    # bc_nanopillar_y = DirichletBC(V1.sub(1), uyFixed, mf_dirichlet, 1)
    # bc_floor = DirichletBC(V1.sub(2), zDisplFloor, mf_dirichlet, 7)
    # bcs = [bc_nanopillar_x, bc_nanopillar_y, bc_nanopillar_z, 
    #         bc_floor, bc_symm1_y, bc_symm2_x]
    # solver = init_solver(Fvar, fmixed, bcs, Jvar)

    if not nuc_dict["nuc_only"]:
        u_start_vec = u_start.vector()[:]
        V1lin_full = u_start.function_space()
        zRoofPM = max(mf3_full.mesh().coordinates()[:,2])
    else:
        zRoofPM = 0
    
    V_vector = FunctionSpace(mesh, VectorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
    V = FunctionSpace(mesh, "P", 1)
    # normal_expr = Expression(("(x[0]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))", 
    #                         "(x[1]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))",
    #                         "((x[2]-b)/pow(b,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))"),
    #                             a=nucRad1, b=nucRad2, degree=1)
    z0 = min(mesh.coordinates()[:,2])
    normal_expr = Expression(("x[0]/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))",
                              "x[1]/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))",
                              "(x[2]-z0)/sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2]-z0,2))"),
                            z0=nucRad2+z0, degree=1)
    normals = project(normal_expr, V_vector)
    outer_area_factor = J * dot(normals, inv(F))

    uEval = fmixed.sub(0)
    uEval.set_allow_extrapolation(True)

    idx += 1
    keepSwimming = True
    it = 0
    pressIts = [pRamp[-1]]
    while keepSwimming:
        # need to keep running the solver until the BCs do not change for the current force
        it += 1
        set_step = False
        # try solving, if diverges, take smaller step
        while not set_step:
            curPress = pRamp[-1]
            curForce = kRamp[-1]
            logger.info(f"Current force is {curForce}")
            logger.info(f"Current pressure is {curPress}")
            forceConst.assign(curForce)
            pressConst.assign(curPress)
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

        if it >= 10:
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
        pressIts.append(pressIts[0] - innerBulkMod*(volLagrange-vol_inner[-1])/volLagrange)
        pRamp[-1] = pressIts[-1]
        if np.abs((pressIts[-1]-pressIts[-2])/pressIts[-2]) > 0.01:
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
            zClampCur = (assemble(uEval[2]*sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1))/
                         assemble(sqrt(inner(outer_area_factor,outer_area_factor))*ds_roofInt(1)))#uEval(0,npRad,z0)[2]
            bc_symm3_z = DirichletBC(V1.sub(2), Constant(zClampCur), mf_dirichlet2, 8)
            if stick:
                bcs = [bc_nanopillar_x, bc_nanopillar_y, bc_nanopillar_z, bc_symm3_z,
                        bc_floor, bc_symm1_y, bc_symm2_x, bc_side_x, bc_side_y]
            else:
                # bcs = [bc_nanopillar_z, bc_floor, bc_symm1_y, bc_symm2_x, bc_side_x, bc_side_y]
                bcs = [bc_floor, bc_symm1_y, bc_symm2_x, bc_symm3_z, bc_side_x, bc_side_y]
            solver = init_solver(Fvar, fmixed, bcs, Jvar)
        
        if keepSwimming:
            logger.info(f"Computing idx = {idx}, outer iteration {it}, def = {uEval(0,0,2*nucRad2)[2]}")
        else:
            logger.info(f"Done computing idx = {idx} after {it} iterations, def = {uEval(0,0,2*nucRad2)[2]}")
        # linFcn_full = Function(V1lin_full)
        # linFcnCoords = V1lin_full.tabulate_dof_coordinates()
        # linFcnVec = linFcn_full.vector()[:]
        # for i in range(0,len(linFcnCoords),3):
        #     xcur = linFcnCoords[i] - u_start_vec[i:i+3]
        #     ucur = uEval(xcur)
        #     linFcnVec[i:i+3] = ucur - u_start_vec[i:i+3] # deformation w.r.t. starting geo in cell
        #     linFcnVec[i+2] += zShift
        # linFcn_full.vector().set_local(linFcnVec)
        # linFcn_full.vector().apply("insert")
        # try:
        #     uLagrangeFull, volLagrange = lagrangeMap(mf3_full, 2, mf2_full, [10,12], 
        #                             [linFcn_full,linFcn_full], 1/8, uLagrangeFull)
        #     volLagrange *= 2
        # except:
        #     print("Could not compute current volume, using previous value")
    zTop = zShift + 2*nucRad2 + uEval(0,0,2*nucRad2)[2]
    zTops.append(zTop)
    if zTop < zRoofPM-dSteric:
        linFcn_full = Function(V1lin_full)
        linFcnCoords = V1lin_full.tabulate_dof_coordinates()
        linFcnVec = linFcn_full.vector()[:]
        for i in range(0,len(linFcnCoords),3):
            xcur = linFcnCoords[i] - u_start_vec[i:i+3]
            ucur = uEval(xcur)
            linFcnVec[i:i+3] = ucur - u_start_vec[i:i+3] # deformation w.r.t. starting geo in cell
            linFcnVec[i+2] += zShift
        linFcn_full.vector().set_local(linFcnVec)
        linFcn_full.vector().apply("insert")
        
        manual_map = False
        # try:
        #     uLagrangeFull, volFull = lagrangeMap(mf3_full, 2, mf2_full, [10,12], 
        #                                 [linFcn_full,linFcn_full], 1/4, uLagrangeFull)
        # except:
        #     print("Failed Lagrange mapping")
        if manual_map:
            mesh_full = uLagrangeFull.function_space().mesh()
            uLagrangeFullCur = sub_to_parent(linFcn_full, mesh_full)
            uLagrangeFull.assign(uLagrangeFullCur)
            ufull_vec = uLagrangeFull.vector()[:]
            ufull_dof = vertex_to_dof_map(uLagrangeFull.function_space())
            closest_ne_indices = closest_ne_indices.astype(int)
            for offset in [0,1,2]:
                curidx = closest_ne_indices[all_cyto_indices]
                ufull_vec[ufull_dof[3*all_cyto_indices+offset]] = scale_vec[all_cyto_indices] * ufull_vec[ufull_dof[3*curidx+offset]]
            
            nuc_mesh = uLagrangeNuc.function_space().mesh()
            nuc_map = np.array(nuc_mesh.topology().mapping()[mesh_full.id()].vertex_map())
            mf3_nuc = MeshFunction("size_t", nuc_mesh, 3, 2)
            mf2_nuc = MeshFunction("size_t", nuc_mesh, 2)
            class allBound(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary
            markBound = allBound()
            markBound.mark(mf2_nuc, 12)
            for f in facets(nuc_mesh):
                theta_cur = np.arctan2(f.midpoint().y(),f.midpoint().x())
                if theta_cur > .2499*np.pi or f.midpoint().y() == 0.0:
                    mf2_nuc[f] = 0

            V1nuc = VectorFunctionSpace(nuc_mesh, "P", 1)
            linFcn_nuc = interpolate(uLagrangeFull, V1nuc)

            uLagrangeNuc, volNuc = lagrangeMap(mf3_nuc, 2, mf2_nuc, [12], 
                                        [linFcn_nuc], 1/8, uLagrangeNuc, degree=1)
            
            nucdof = dof_to_vertex_map(V1nuc)
            nuc_xindices = ufull_dof[3*nuc_map[np.floor(nucdof[::3]/3).astype(int)]]
            nuc_yindices = nuc_xindices + 1
            nuc_zindices = nuc_xindices + 2
            ufull_vec[nuc_xindices] = uLagrangeNuc.vector()[::3]
            ufull_vec[nuc_yindices] = uLagrangeNuc.vector()[1::3]
            ufull_vec[nuc_zindices] = uLagrangeNuc.vector()[2::3]
            uLagrangeFull.vector().set_local(ufull_vec)
            uLagrangeFull.vector().apply("insert")

        else:
            uLagrangeFull, volFull = lagrangeMap(mf3_full, 2, mf2_full, [10,12], 
                                        [linFcn_full,linFcn_full], 1/4, uLagrangeFull)

        if "tvec" in nuc_dict.keys():
            uFull_file.write(uLagrangeFull, nuc_dict["tvec"][-1])
        else:
            uFull_file.write(uLagrangeFull, idx)
        # vol_inner.append(volLagrange)
        # pNew = pRamp[-1] - 200*(vol_inner[-1]-vol_inner[-2])/vol_inner[-1]
    # else:
    #     raise ValueError("Nucleus should always stay below top of PM in this case")

    # set next pressure values
    linFcn = interpolate(fmixed.sub(0), V1lin)
    try:
        uLagrange, volLagrange = lagrangeMap(mf3, 2, mf2, [10,12], 
                                    [linFcn,linFcn], 1/4, uLagrange)
    except:
        logger.warning("Current volume could not be solved for")
    vol_inner.append(volLagrange)
    pNew = pRamp[-1] - innerBulkMod*(vol_inner[-1]-vol_inner[-2])/vol_inner[-1]

    u_file.write_checkpoint(fmixed.sub(0), "u_np_ellipsoid", idx, append=True)
    ulin.assign(project(fmixed.sub(0), V_vector))
    ulin_file.write(ulin, idx)

    if len(fmixed.sub(0).vector()) == len(u_prev.vector()):
        u_prev.vector()[:] = fmixed.sub(0).vector()[:]
        u_prev.vector().apply("insert")
    else:
        raise ValueError("Could not reassign u_prev!!")
    
    # if len(fmixed.sub(2).vector()) == len(psi_c_prev.vector()):
    #     psi_c_prev.vector()[:] = fmixed.sub(2).vector()[:]
    #     psi_c_prev.vector().apply("insert")
    # else:
    #     raise ValueError("Could not reassign psi_c_prev!!")

    # zNP.append((zNP[-1]+u(0,0,0)[2])/2)
    # kRepel += 0.01

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
    surf_stress.assign(project(dot(normals, Ttensor), V_vector))
    surf_stress_file.write(surf_stress, idx)
    psi_fcn.assign(project(psi, V_scalar))
    psi_file.write(psi_fcn, idx)

    if not nuc_dict["nuc_only"]:
        a_vector.set_allow_extrapolation(True)
        aFullVec = aLagrangeFull.vector()[:]
        aInnerVec = aLagrangeInner.vector()[:]
        uInnerVec = uLagrangeInner.vector()[:]
        u_start_inner_vec = u_start_inner.vector()[:]
        for i in range(0,len(linFcnCoords),3):
            xcur = linFcnCoords[i] - u_start_vec[i:i+3]
            acur = a_vector(xcur)
            aFullVec[i:i+3] = acur
            xcur_inner = linFcnCoords[i] - u_start_inner_vec[i:i+3]
            # rcur = np.sqrt(xcur[0]**2 + xcur[1]**2 + (xcur[2]-nucRad2)**2)
            # if np.abs(rcur - nucRad1) > 0.01*nucRad1:
            #     raise ValueError("Not at the right point")
            # if nucRad1 != nucRad2 or rthickness != zthickness:
            #     raise ValueError("inner extrapolation does not work for this geo")
            # xcur_inner = xcur * (nucRad2-rthickness)/nucRad2
            # xcur_inner[2] = nucRad2 + (xcur[2]-nucRad2) * (nucRad2-rthickness)/nucRad2
            acur_inner = a_vector(xcur_inner)
            ucur_inner = uEval(xcur_inner)
            aInnerVec[i:i+3] = acur_inner
            uInnerVec[i:i+3] = ucur_inner - u_start_inner_vec[i:i+3]
            uInnerVec[i+2] += zShift
        aLagrangeFull.vector().set_local(aFullVec)
        aLagrangeFull.vector().apply("insert")
        aLagrangeInner.vector().set_local(aInnerVec)
        aLagrangeInner.vector().apply("insert")
        uLagrangeInner.vector().set_local(uInnerVec)
        uLagrangeInner.vector().apply("insert")
        aFull_file.write(aLagrangeFull, idx)
        aInner_file.write(aLagrangeInner, idx)
        uInner_file.write(uLagrangeInner, idx)

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

    if not nuc_dict["nuc_only"]:
        nuc_dict = {"idx": idx, "pRamp": pRamp, "kRamp": kRamp, "forceConst": forceConst, "kInc": kInc,
                    "pressConst": pressConst, "fmixed": fmixed, "solver": solver, "Fvar": Fvar,
                    "Jvar": Jvar, "bcs": bcs, "mf_dirichlet": mf_dirichlet, "mf_dirichlet2": mf_dirichlet2, "domain_id": domain_id, 
                    "mf_surf": mf_surf, "nanopillar_specs": nanopillar_specs, "uxFixed": uxFixed, "uyFixed": uyFixed,
                    "u_start": u_start, "u_start_inner": u_start_inner, "mf3_full": mf3_full, "mf2_full": mf2_full, "vol_inner": vol_inner,
                    "nucRad1": nucRad1, "nucRad2": nucRad2, "zDisplOuter": zDisplOuter, 
                    "dSteric": dSteric, "u_prev": u_prev, "a_vector": a_vector, "u_file": u_file, "a_file": a_file, 
                    "uFull_file": uFull_file, "aFull_file": aFull_file, "aInner_file": aInner_file, "uInner_file": uInner_file,
                    "results_folder": results_folder, "aLagrangeFull": aLagrangeFull, "aLagrangeInner": aLagrangeInner, 
                    "uLagrangeFull": uLagrangeFull, "uLagrangeInner": uLagrangeInner, "inner_SA": inner_SA, "outer_SA": outer_SA, "vol": vol, 
                    "volLagrange": volLagrange, "rthickness": rthickness, "zthickness": zthickness, 
                    "fmixed_prev": fmixed_prev, "mf3": mf3, "mf2": mf2, "uLagrange": uLagrange,
                    "closest_ne_indices": closest_ne_indices, "all_cyto_indices": all_cyto_indices, 
                    "scale_vec": scale_vec, "uLagrangeNuc": uLagrangeNuc, "innerBulkMod": innerBulkMod,
                    "ulin": ulin, "ulin_file": ulin_file, "zTops": zTops, "nuc_only": nuc_dict["nuc_only"],
                    "surf_stress": surf_stress, "surf_stress_file": surf_stress_file, "psi_fcn": psi_fcn, "psi_file": psi_file,
                    "E1scale": E1scale, "E2scale": E2scale, "psi_c_prev": psi_c_prev, "ds_roofInt": ds_roofInt}
    else:
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
                    "surf_stress": surf_stress, "surf_stress_file": surf_stress_file, "psi_fcn": psi_fcn, "psi_file": psi_file,
                    "E1scale": E1scale, "E2scale": E2scale, "psi_c_prev": psi_c_prev, "ds_roofInt": ds_roofInt}
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
    V_scalar = uxFixed.function_space()
    ulin = project(uEval, V_vector)
    ulin.set_allow_extrapolation(True)
    utest = ulin.copy(deepcopy=True)
    testJacobian = project(det(Identity(3) + grad(ulin)), V_scalar)

    badJacobian = MeshFunction("size_t", mesh, 2, 0)
    # for c in cells(mesh):
    #     Jvals = testJacobian.vector()[dofmap_x[c.entities(0)]]
    #     if np.any(Jvals < 0.0):
    #         for f in facets(c):
    #             badJacobian[f] = 1

    for f in facets(mesh):
        if mf_surf[f] == 10:
            xCur = [f.midpoint().x(), f.midpoint().y(), f.midpoint().z()]
            uCur = uEval(xCur)
            xDef = xCur + uCur
            all_dist = np.sqrt((xNP-xDef[0])**2 + (yNP-xDef[1])**2)
            if xDef[2] <= 2.0:#1e-6:
                if np.any(all_dist <= npRad) or badJacobian[f] == 1 and False:
                    if mf_dirichlet[f] != 1:# and np.sqrt(xDef[0]**2+xDef[1]**2) <= 0.0:# 0.8*npRad:
                        # if xDef[2] < -0.2+1e-6:
                        #     print("This could be an issue!!")
                        #     continue
                        reinit = True
                        keepSwimming = True
                        mf_dirichlet[f] = 1
                        curNodes = f.entities(0)
                        curCoords = mesh.coordinates()[curNodes,:]
                        for n in range(len(curNodes)):
                            uCur = uEval(curCoords[n,:])
                            uxFixed.vector()[dofmap_x[curNodes[n]]] = uCur[0]
                            uyFixed.vector()[dofmap_y[curNodes[n]]] = uCur[1]
                        for u in [uxFixed,uyFixed]:
                            u.vector().apply("insert")
                        # if xDef[0] > 1.8:
                        #     print("Pause for case")
                        domain_id[f] = 11
                elif np.any(all_dist <= npRad+0.2) and xDef[2] < -100:#0.5+1e-6:
                    print('On side of nanopillar here')
                    closest_idx = np.argmin(all_dist)
                    if mf_dirichlet[f] != 8:
                        reinit = True
                        keepSwimming = True
                        mf_dirichlet[f] = 8
                        curNodes = f.entities(0)
                        curCoords = mesh.coordinates()[curNodes,:]
                        for n in range(len(curNodes)):
                            uCur = uEval(curCoords[n,:])
                            xTest = curCoords[n,0] + uCur[0] - xNP[closest_idx]
                            yTest = curCoords[n,1] + uCur[1] - yNP[closest_idx]
                            rTest = np.sqrt(xTest**2 + yTest**2)
                            xFix = (npRad/rTest) * xTest
                            yFix = (npRad/rTest) * yTest
                            uxFixed.vector()[dofmap_x[curNodes[n]]] = uCur[0] + (xFix-xTest)
                            uyFixed.vector()[dofmap_y[curNodes[n]]] = uCur[1] + (yFix-yTest)
                        for u in [uxFixed,uyFixed]:
                            u.vector().apply("insert")
                        # if xDef[0] > 1.8:
                        #     print("Pause for case")
                        domain_id[f] = 11
                elif mf_dirichlet[f] == 1 and not np.any(all_dist <= npRad) and not stick:# or xDef[2] <= (-hNP+1e-6):
                    reinit = True
                    keepSwimming = True
                    mf_dirichlet[f] = 0
                    domain_id[f] = 1
                elif xDef[2] <= (-hNP+1e-6) and mf_dirichlet[f] != 7:
                    reinit = True
                    keepSwimming = True
                    mf_dirichlet[f] = 7
                    # domain_id[f] = 11
                elif mf_dirichlet[f] == 7 and not (xDef[2] <= (-hNP+1e-6)): # should only be if hNP is increasing
                    reinit = True
                    keepSwimming = True
                    mf_dirichlet[f] = 0
                    domain_id[f] = 10
            # elif mf_dirichlet[f] == 1 and not np.any(all_dist <= npRad):# or xDef[2] <= (-hNP+1e-6):
            #     reinit = True
            #     keepSwimming = True
            #     mf_dirichlet[f] = 0
            #     domain_id[f] = 1
            elif xDef[2] >= zRoof and mf_dirichlet[f] != 4:
                domain_id[f] = 1
                reinit = True
                mf_dirichlet[f] = 4
                keepSwimming = True
        
    if reinit:
        logger.debug("Need to reinit")
    return (reinit, keepSwimming, mf_dirichlet, domain_id, uxFixed, uyFixed)
