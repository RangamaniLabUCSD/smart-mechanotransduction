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
smart_logger.setLevel(logging.DEBUG)
logger = logging.getLogger("mechanotransduction")
logger.setLevel(logging.INFO)
logger.info("Starting nuclear mechanics example")

# here = pathlib.Path.cwd() 
sys.path.insert(0, (here / ".." / "mesh-files").as_posix())
import spread_cell_mesh_generation_old as mesh_gen

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

parser = argparse.ArgumentParser()
add_nucmech_arguments(parser)
args = vars(parser.parse_args())
run_local = True
if run_local:
    cur_dir = str(pathlib.Path.cwd() / "..")
    args["mesh_folder"] = pathlib.Path("")
    args["max_force"] = 100.0
    args["start_force"] = 0.0
    args["u0"] = pathlib.Path("")
    args["bulk_mod"] = 1e8
    args["nanopillar_radius"] = 0.5
    args["nanopillar_height"] = 0.0
    args["nanopillar_spacing"] = 3.0
    args["outdir"] = pathlib.Path(f"/root/shared/gitrepos/nuc_indent_testOldScript")#{args['bulk_mod']}J"
                                #   f"_npRad{args['nanopillar_radius']}"
                                #   f"_npSpacing{args['nanopillar_spacing']}_5layersWithPress")

# Create mesh and define function space
nucRad1 = 5.09#6.625#5.5
nucRad2 = 5.09#3.0
thickness = 0.05
rthickness = thickness
zthickness = thickness
NE_layers = 2
hEdge = 0.3

zRoof = np.inf
zRoofAlt = 10.0

mesh_ref, mf2, mf3 = mesh_gen.NE_mesh(rRad=nucRad1, zRad=nucRad2, thickness=[rthickness,zthickness],
                                      hEdge=hEdge, hInnerEdge=hEdge, sym_fraction=0.25, NE_layers=NE_layers)        
mesh = create_meshview(mf3, 1)

# def mf_to_submf(mf_old, mesh):
#     cur_dim = mf_old.dim()
#     parent_mesh = mf_old.mesh()
#     mf_new = MeshFunction("size_t", mesh, cur_dim, 0)
#     if cur_dim == 0:
#         full_map = mesh.topology().mapping()[parent_mesh.id()].vertex_map()
#         mf_new.array()[:] = mf_old.array()[full_map]
#     elif cur_dim == 3:
#         full_map = mesh.topology().mapping()[parent_mesh.id()].cell_map()
#         mf_new.array()[:] = mf_old.array()[full_map]
#     elif cur_dim == 1: # no map exists yet, need to manually map
#         vertex_map = mesh.topology().mapping()[parent_mesh.id()].vertex_map()
#         parent_entities = []
#         for e in edges(parent_mesh):
#             parent_entities.append(e.entities(0))
#         parent_entities = np.array(parent_entities)
#         array_new = mf_new.array()[:]
#         array_old = mf_old.array()[:]
#         full_map = []
#         for e in edges(mesh):
#             full_map.append(find_facet[0][0])
#             cur_entities = e.entities(0)
#             map_entities = vertex_map[cur_entities]
#             find_edge = np.nonzero(np.all(parent_entities==map_entities, axis=1))
#             if len(find_edge[0]) == 1:
#                 array_new[e.index()] = array_old[find_edge[0][0]]
#             else:
#                 raise ValueError("Submesh coords do not match parent coords")
#     elif cur_dim == 2: # no map exists yet, need to manually map
#         vertex_map = np.array(mesh.topology().mapping()[parent_mesh.id()].vertex_map())
#         parent_entities = []
#         for f in facets(parent_mesh):
#             parent_entities.append(f.entities(0))
#         parent_entities = np.array(parent_entities)
#         array_new = mf_new.array()[:]
#         array_old = mf_old.array()[:]
#         full_map = []
#         for f in facets(mesh):
#             print(f"Assessing facet {f.index()} of {mesh.num_facets()}")
#             cur_entities = f.entities(0)
#             map_entities = vertex_map[cur_entities]
#             find_facet = np.nonzero(np.all(parent_entities==map_entities, axis=1))
#             full_map.append(find_facet[0][0])
#             if len(find_facet[0]) == 1:
#                 array_new[f.index()] = array_old[find_facet[0][0]]
#             else:
#                 raise ValueError("Submesh coords do not match parent coords")

#     return mf_new, full_map

# mf_surf, facet_map = mf_to_submf(mf2, mesh)
# define nanopillar dimensions for current case
npSpacing = args["nanopillar_spacing"]
npRad = args["nanopillar_radius"]
hNP = args["nanopillar_height"]
xMax = np.ceil((max([nucRad1,nucRad2])) / npSpacing) * npSpacing
xNP = np.arange(-xMax, xMax+1e-12, npSpacing)
yNP = np.arange(-xMax, xMax+1e-12, npSpacing)
xNP, yNP = np.meshgrid(xNP, yNP)
xNP = xNP.flatten()
yNP = yNP.flatten()

mf_surf = MeshFunction("size_t", mesh, 2, 0)
mf_dirichlet = MeshFunction("size_t", mesh, 2, 0)
mf_edge = MeshFunction("size_t", mesh, 2, 0)

# redetermine surface markers
# rad_eff1 = (nucRad1**2 * nucRad2)**(1/3)
# rad_eff2 = ((nucRad1-rthickness)**2 * (nucRad2-zthickness))**(1/3)
# thickness_thresh = ((rad_eff1 - rad_eff2) / NE_layers)
# class OuterSurf(SubDomain):
#     def inside(self, x, on_boundary):
#         bound_val = pow(pow(x[0]/nucRad1,2) + pow(x[1]/nucRad1,2) + 
#                     pow((x[2]-nucRad2)/nucRad2,2),0.5)
#         cutoffFrac = 1-0.5*thickness_thresh/rad_eff1
#         return bound_val > cutoffFrac and on_boundary
# class InnerSurf(SubDomain):
#     def inside(self, x, on_boundary):
#         bound_val = pow(pow(x[0]/(nucRad1-rthickness),2) + pow(x[1]/(nucRad1-rthickness),2) +
#                     pow((x[2]-nucRad2)/(nucRad2-zthickness),2),0.5)
#         cutoffFrac1 = 1+0.5*thickness_thresh/rad_eff2
#         cutoffFrac2 = 1-0.5*thickness_thresh/rad_eff2
#         return  bound_val < cutoffFrac1 and bound_val > cutoffFrac2
class OuterSurf(SubDomain):
    def inside(self, x, on_boundary):
        return pow(pow(x[0]/nucRad1,2) + pow(x[1]/nucRad1,2) + 
                   pow((x[2]-nucRad2)/nucRad2,2),0.5) > 0.999 and on_boundary
        # return on_boundary and near(x[2], 0.0)
class InnerSurf(SubDomain):
    def inside(self, x, on_boundary):
        return pow(pow(x[0]/(nucRad1-thickness),2) + pow(x[1]/(nucRad1-thickness),2) +
                   pow((x[2]-nucRad2)/(nucRad2-thickness),2),0.5) < 1.001 and on_boundary
        # return on_boundary and near(x[2], thickness)

class NanopillarContact(SubDomain):
    def inside(self, x, on_boundary):
        radialVal = np.sqrt((x[0]/(nucRad1))**2 + (x[1]/(nucRad1))**2 +
                            ((x[2]-nucRad2)/(nucRad2))**2)
        return (np.sqrt(x[0]**2 + x[1]**2) < npRad and x[2] < nucRad2 and on_boundary
                and radialVal > 0.999)
        # return on_boundary and near(x[2], 0.0) and (np.sqrt(x[0]**2 + x[1]**2) < 0.2)
class NanopillarContactInner(SubDomain):
    def inside(self, x, on_boundary):
        radialVal = np.sqrt((x[0]/(nucRad1-rthickness))**2 + (x[1]/(nucRad1-rthickness))**2 +
                            ((x[2]-nucRad2)/(nucRad2-zthickness))**2)
        return (np.sqrt(x[0]**2 + x[1]**2) < 1*npRad and x[2] < nucRad2 and on_boundary
                and radialVal < 1.001)
class UpperContact(SubDomain):
    def inside(self, x, on_boundary):
        radialVal = np.sqrt((x[0]/(nucRad1))**2 + (x[1]/(nucRad1))**2 +
                            ((x[2]-nucRad2)/(nucRad2))**2)
        return (x[2] > zRoof and radialVal > 0.999)
class UpperContactInner(SubDomain):
    def inside(self, x, on_boundary):
        radialVal = np.sqrt((x[0]/(nucRad1-rthickness))**2 + (x[1]/(nucRad1-rthickness))**2 +
                            ((x[2]-nucRad2)/(nucRad2-zthickness))**2)
        return (x[2] > zRoof-zthickness and radialVal < 1.001)
class SymmAxis1(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.001 and on_boundary
class SymmAxis2(SubDomain):
    def inside(self, x, on_boundary):
        # return np.arctan(x[1]/x[0]) > 0.999*np.pi/4 and on_boundary
        return x[0] < 0.001 and on_boundary

outerSurf = OuterSurf()
innerSurf = InnerSurf()
outerSurf.mark(mf_surf, 10) # mark outer surface points as 10
innerSurf.mark(mf_surf, 12) # mark inner surface points as 12

nanopillar = NanopillarContact()
nanopillarInner = NanopillarContactInner()
roof = UpperContact()
roofInner = UpperContactInner()
symm1 = SymmAxis1()
symm2 = SymmAxis2()
nanopillar.mark(mf_dirichlet, 1) # mark nanopillar contact with 1
nanopillarInner.mark(mf_dirichlet, 5)
array_dirichlet = mf_dirichlet.array()[:]
array_ref = mf_surf.array()[:]
array_dirichlet[np.logical_and(array_dirichlet == 1, array_ref != 10)] = 0

roof.mark(mf_dirichlet, 4)
roofInner.mark(mf_dirichlet, 6)
symm1.mark(mf_dirichlet, 2)
symm2.mark(mf_dirichlet, 3)

# symmEdge = SymmEdge()
# symmEdge.mark(mf_edge, 1)
# mf_bound = MeshFunction("size_t", mesh, 2, 0)
# class MarkBound(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary
# bound = MarkBound()
# bound.mark(mf_bound, 1)

mf_vol = MeshFunction("size_t", mesh, 3, 1)
# class InnerVol(SubDomain):
#     def inside(self, x, on_boundary):
#         # return np.arctan(x[1]/x[0]) > 0.999*np.pi/4 and on_boundary
#         return pow(pow(x[0]/(nucRad1-rthickness),2) + pow(x[1]/(nucRad1-rthickness),2) +
#                    pow((x[2]-nucRad2)/(nucRad2-zthickness),2),0.5) < 0.999
    
# inner_vol = InnerVol()
# inner_vol.mark(mf_vol, 2)
# mf_vol, cell_map = mf_to_submf(mf3, mesh)

V_vector = FunctionSpace(mesh, VectorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
# V_tensor = FunctionSpace(mesh, TensorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
V_scalar = FunctionSpace(mesh, "P", 1)
normal_expr = Expression(("(x[0]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))", 
                          "(x[1]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))",
                          "((x[2]-b)/pow(b,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))"),
                            a=nucRad1, b=nucRad2, degree=1)
normals = project(normal_expr, V_vector)

inner_normal_expr = Expression(("(x[0]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))", 
                          "(x[1]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))",
                          "((x[2]-b)/pow(b,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))"),
                            a=nucRad1-rthickness, b=nucRad2-zthickness, degree=1)
inner_normals = project(normal_expr, V_vector)

# mesh = RectangleMesh(Point(0.0, 0.0), Point(1.5*nucRad1, thickness), 151, 5)

x = SpatialCoordinate(mesh)
# results_folder = "/root/scratch/nuc_indent_1e6J_eighth"
results_folder = args["outdir"]
File(f"{results_folder}/test_shell.pvd") << mesh

# Define mixed function space for displacements over each region
el1 = VectorElement("P", mesh.ufl_cell(), 2) # u function space
el2 = FiniteElement("P", mesh.ufl_cell(), 1) # p function space
mixed_element = MixedElement([el1, el2]) # mixed function space
Vmixed = FunctionSpace(mesh, mixed_element)
V1 = Vmixed.sub(0)

# Mark boundary subdomains
# fixed_bound1 = CompiledSubDomain("(on_boundary && x[1] < side1) && x[0] < side2", side1 = nucRad2, side2 = 0.01*nucRad1)
# fixed_bound2 = CompiledSubDomain("(on_boundary && x[1] < side1) && (x[0] < 6*side2 && x[0] > 5*side2)", side1 = nucRad2, side2 = 0.1*nucRad1)
# left = CompiledSubDomain("near(x[0], 0.0) && on_boundary")

# Define Dirichlet boundary conditions
Vouter = FunctionSpace(create_meshview(mf_surf, 10), "P", 1)
Vinner = FunctionSpace(create_meshview(mf_surf, 12), "P", 1)
# zDispl = Expression(("0.0","0.0","-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))"), degree=1, r1=nucRad1, r2=nucRad2)
zDisplOuter = Expression("-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, r1=nucRad1, r2=nucRad2)
# zDisplOuter = interpolate(zDisplOuterExpr, Vouter)
zDisplInner = Expression("z1-(r2-(r2-z1)*sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, 
                         z1=zthickness, r1=nucRad1-rthickness, r2=nucRad2)
zDisplUpper = Expression("zMax-r2*(1+sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, zMax=zRoof, r1=nucRad1, r2=nucRad2)
zDisplUpperInner = Expression("zMax-(r2+(r2-z1)*sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, 
                              zMax=zRoof-zthickness, r1=nucRad1-rthickness, r2=nucRad2, z1=zthickness)
# zDispl = Expression("0.0", degree=1)
bc_nanopillar = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
bc_nanopillar_inner = DirichletBC(V1.sub(2), zDisplInner, mf_dirichlet, 5)
bc_upper = DirichletBC(V1.sub(2), zDisplUpper, mf_dirichlet, 4)
bc_upper_inner = DirichletBC(V1.sub(2), zDisplUpperInner, mf_dirichlet, 6)
# bc_fixed2 = DirichletBC(V1.sub(1), zDispl, fixed_bound2)
bc_symm1_x = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 2)
bc_symm1_y = DirichletBC(V1.sub(1), Constant(0.0), mf_dirichlet, 2)
bc_symm2_x = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 3)
bc_symm2_y = DirichletBC(V1.sub(1), Constant(0.0), mf_dirichlet, 3)
# bc_symmEdge1 = DirichletBC(V1.sub(0), Constant(0.0), mf_edge, 1)
# bc_symmEdge2 = DirichletBC(V1.sub(1), Constant(0.0), mf_edge, 1)
if not np.isinf(zRoof):
    bcs = [bc_nanopillar, bc_nanopillar_inner, bc_upper, 
           bc_upper_inner, bc_symm1_y, bc_symm2_x]
else:
    bcs = [bc_nanopillar, bc_nanopillar_inner, bc_symm1_y, bc_symm2_x]

# Define functions
# du = TrialFunction(V1)
# v = TestFunction(V1)
# u = Function(V1)
(du, dp) = TrialFunctions(Vmixed)            # Incremental displacement and dp
(v, q)  = TestFunctions(Vmixed)             # Test function
fmixed  = Function(Vmixed)                 # Displacement from previous iteration
(u, p) = split(fmixed)

domain_id = MeshFunction("size_t", mesh, 2, 0)
for f in facets(mesh):
    if mf_dirichlet[f] == 2:
        domain_id[f] = 2
    elif mf_dirichlet[f] == 3:
        domain_id[f] = 3
    elif f.midpoint().z() > 1.2*nucRad2 and mf_surf[f] == 10:
        domain_id[f] = 1
    elif f.midpoint().z() > 1.2*nucRad2 and mf_surf[f] == 12:
        domain_id[f] = 4
    elif mf_dirichlet[f] == 1:
        domain_id[f] = 11
    elif mf_surf[f] == 10:
        domain_id[f] = 10
    elif mf_surf[f] == 12:
        domain_id[f] = 12
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

# Invariants of deformation tensors
I1 = variable(tr(C))
I2 = variable(0.5*(I1**2 - tr(C*C)))
J  = variable(det(F))

# rescale I1, I2
# I1 = I1 * J**(-2/3)
# I2 = I2 * J**(-4/3)

# Elasticity parameters
E1, E2 = Constant(5000.0), Constant(1000.0)

# Stored strain energy density (incompressible Mooney Rivlin model)
# psi = (mu/2)*(I1 - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
psi = E1*(I1-3) + E2*(I2-3) - p*(J-1)
psi2 = 0.1*psi
# psi = E1*(I1-3) + 4e4*(J-1)**2

zNP = [-0.05]#[-nucRad - 0.05]
idx = 0
zMove = 0.1
zStep = 0.1
zFinal = zNP[-1] + zMove
kMin = args["start_force"]
kMax = args["max_force"]
kInc = 0.5 #max([0.001, (args["max_force"]-args["start_force"])/1000])
kRamp = [0.0]#[kMin]
pMin = 0.0
pMax = 140.0
pRamp = [pMin]
p0 = Constant(0e4)
dpress = 20.0

u_file = XDMFFile(f"{results_folder}/u_np_ellipsoid.xdmf")
u_file.parameters["flush_output"] = True
u_file.write(fmixed.split()[0], pRamp[-1]+kRamp[-1])
a_vector = J * dot(normals, inv(F))
a_vector = project(a_vector, V_vector)
a_file = XDMFFile(f"{results_folder}/a_np_ellipsoid.xdmf")
a_file.parameters["flush_output"] = True
a_file.write(a_vector, pRamp[-1]+kRamp[-1])

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

curForce = 0.0
curPress = pRamp[-1]
print(f"Starting force is {curForce}")
print(f"Starting pressure is {curPress}")
forceConst = Constant(curForce)
pressConst = Constant(curPress)
Tdir = Expression(("0.0", "0.0", "-1.0"), degree=1)
# Pinner = Expression(("0.0", "pressure"), degree=1, pressure=450)
outer_area_factor = J * dot(normals, inv(F))
inner_area_factor = J * dot(inner_normals, inv(F))
T = forceConst*Tdir*sqrt(inner(outer_area_factor,outer_area_factor))
Press_in = (p0 + pressConst)*inner_area_factor
Press_out = p0 * outer_area_factor
topContactForce = Function(V_vector)
topContactForce_inner = Function(V_vector)

# Convert potential energy to first Piola-Kirchoff stress tensor
dx = Measure("dx", domain=mesh, subdomain_data=mf_vol)
Ttensor = diff(psi, F)
Ttensor_in = diff(psi2, F)
# Pi = ufl.diff(psi, F) + p * J * ufl.inv(F.T)
# Pi = psi*x[0]*dx - dot(T,u)*x[0]*ds(1) #+ dot(Ptop,u)*x[0]*ds(2)
# Fvar = derivative(Pi, u, v) #derivative(Pi, u, v) + derivative(Pi, p, q)
# Fvar = derivative(Pi, u, v) - q*(J - 1)*x[0]*dx
Fvar = (inner(grad(v), Ttensor)*dx(1) + inner(grad(v), Ttensor_in)*dx(2) - 
        inner(v, T+Press_out)*ds(1) - inner(v, topContactForce)*ds(1) - inner(v, topContactForce_inner+Press_in)*ds(4) - 
        inner(v, Press_in)*ds(12) - inner(v, Press_out)*ds(10) + 
        inner(q, args["bulk_mod"]*(J-1) + p) * dx(1) + inner(q, 0.1*args["bulk_mod"]*(J-1) + p) * dx(2))
        # + inner(10e3*(u-u_prev), v) * ds(11)) # positive for "extra slip"
test_var = u - u_prev
# + inner(q, u[0]-u[1])*ds(3) + inner(q, u[1])*ds(2)
# Compute Jacobian of F
dw = TrialFunction(Vmixed)
Jvar = derivative(Fvar, fmixed)#, dw)
# du0, du1, du2 = split(du)
# u0, u1, u2 = split(u)
# Jvar = (derivative(Fvar, u0, du0) + derivative(Fvar, u0, du1) + derivative(Fvar, u0, du2) + derivative(Fvar, u0, dp) +
#         derivative(Fvar, u1, du0) + derivative(Fvar, u1, du1) + derivative(Fvar, u1, du2) + derivative(Fvar, u1, dp) +
#         derivative(Fvar, u2, du0) + derivative(Fvar, u2, du1) + derivative(Fvar, u2, du2) + derivative(Fvar, u2, dp) +
#         derivative(Fvar, p, du0) + derivative(Fvar, p, du1) + derivative(Fvar, p, du2) + derivative(Fvar, p, dp))
# Define problem and solver with custom settings
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

# def init_custom_solver(Fvar, fmixed, bcs, Jvar):

#     class Problem(NonlinearProblem):
#         def __init__(self, J, F, bcs):
#             self.bilinear_form = J
#             self.linear_form = F
#             self.bcs = bcs
#             NonlinearProblem.__init__(self)

#         def F(self, b, x):
#             assemble(self.linear_form, tensor=b)
#             for bc in self.bcs:
#                 bc.apply(b, x)

#         def J(self, A, x):
#             assemble(self.bilinear_form, tensor=A)
#             for bc in self.bcs:
#                 bc.apply(A)


#     class CustomSolver(PETScSNESSolver):
#         def __init__(self):
#             PETScSNESSolver.__init__(self, mesh.mpi_comm())

#         def solver_setup(self, A, P, problem, iteration):
#             self.linear_solver().set_operator(A)

#             prm = self.parameters
#             prm["absolute_tolerance"] = 1E-8
#             prm["relative_tolerance"] = 1E-6
#             prm["maximum_iterations"] = 100
#             prm["linear_solver"] = 'bcgs'
#             prm["preconditioner"] = "hypre"
#             prm["line_search"] = "l2"
    # problem = Problem(Jvar, Fvar, bcs)
    # solver = CustomSolver()
    # return (problem, solver)
    # Define the function/jacobian blocks
    
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

custom_solver = False
if custom_solver:
    problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
    # opts = PETSc.Options()
    # opts["snes_linesearch_type"] = "l2"
    # solver.setFromOptions()
else:
    solver = init_solver(Fvar, fmixed, bcs, Jvar)


fmixed_prev = fmixed.vector()[:].copy()

while kRamp[-1] < kMax: #min(u.vector()[V1.dofmap().dofs()[2:-1:3]]) > -zIndentMax:

    idx += 1
    keepSwimming = True
    it = 0
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
            print(f"Current force is {curForce}")
            print(f"Current pressure is {curPress}")
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
                    solver.solve()
                    set_step = True
                # solver.solve(problem, fmixed.vector())
            except:
                reset = True
            if reset:
                if pumpedUp:
                    print(f"Resetting kRamp from {kRamp[-1]} to")
                    if len(kRamp) == 1:
                        kRamp[-1] = kRamp[-1]/2
                    else:
                        kRamp[-1] = (kRamp[-1]+kRamp[-2])/2
                        kInc = kRamp[-1] - kRamp[-2]
                    print(f"{kRamp[-1]} because solve failed")
                    # reset fmixed and reinit solver
                    fmixed.vector()[:] = fmixed_prev
                    if custom_solver:
                        problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
                    else:
                        solver = init_solver(Fvar, fmixed, bcs, Jvar)
                else:
                    print(f"Resetting pRamp from {pRamp[-1]} to")
                    if len(pRamp) == 1:
                        pRamp[-1] = pRamp[-1]/2
                    else:
                        pRamp[-1] = (pRamp[-1]+pRamp[-2])/2
                        dpress = pRamp[-1] - pRamp[-2]
                    print(f"{pRamp[-1]} because solve failed")
                    # reset fmixed and reinit solver
                    fmixed.vector()[:] = fmixed_prev
                    if custom_solver:
                        problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
                    else:
                        solver = init_solver(Fvar, fmixed, bcs, Jvar)

        if it >= 10:
            print(f"Done computing idx = {idx} after maximum ({it}) iterations, def = {uEval(0,0,2*nucRad2)[2]}")
            break
        fmixed_prev = fmixed.vector()[:].copy()
        # update bcs as needed
        coords = mf_dirichlet.mesh().coordinates()
        reinit = False
        keepSwimming = False
        uEval = fmixed.sub(0)
        uEval.set_allow_extrapolation(True)
        for f in facets(mesh):
            if mf_surf[f] == 10:
                xCur = [f.midpoint().x(), f.midpoint().y(), f.midpoint().z()]
                uCur = uEval(xCur)
                xDef = xCur + uCur
                all_dist = np.sqrt((xNP-xDef[0])**2 + (yNP-xDef[1])**2)
                if xDef[2] <= 1e-6 and (np.any(all_dist <= npRad) or xDef[2] <= (-hNP+1e-6)):
                    if mf_dirichlet[f] == 0: 
                        reinit = True
                        keepSwimming = True
                        mf_dirichlet[f] = 1
                        domain_id[f] = 11
                elif mf_dirichlet[f] == 1 and not (np.any(all_dist <= npRad) or xDef[2] <= (-hNP+1e-6)):
                    reinit = True
                    keepSwimming = True
                    mf_dirichlet[f] = 0
                    domain_id[f] = 10
                elif xDef[2] >= zRoof and mf_dirichlet[f] != 4:
                    domain_id[f] = 1
                    reinit = True
                    mf_dirichlet[f] = 4
                    keepSwimming = True
                        # bc_num = len(bcs) + 1
                        # mf_dirichlet[f] = bc_num
                        # bcs.append(DirichletBC(V1, Constant((uCur[0], uCur[1], -xCur[2])), mf_dirichlet, bc_num))
                # if xDef[2] <= 0 and ((rDef <= 0.1*nucRad1)):# or (xDef[0] <= 0.6*nucRad1 and xDef[0] > 0.5*nucRad1)):
                #     mf_dirichlet[f] = 1
                # test for intersections
                # Jtest = project(J, V_scalar)
                # if np.any(Jtest.vector()[:] < 0.0):
                #     print("uh oh!!")
            if mf_surf[f] == 12:
                xCur = [f.midpoint().x(), f.midpoint().y(), f.midpoint().z()]
                uCur = uEval(xCur)
                xDef = xCur + uCur
                all_dist = np.sqrt((xNP-xDef[0])**2 + (yNP-xDef[1])**2)
                if xDef[2] <= zthickness+1e-6 and (np.any(all_dist <= npRad) or xDef[2] <= (-hNP+1e-6+zthickness)):
                    if mf_dirichlet[f] == 0: 
                        reinit = True
                        keepSwimming = True
                        mf_dirichlet[f] = 5
                elif mf_dirichlet[f] == 5 and not (np.any(all_dist <= npRad) or xDef[2] <= (-hNP+1e-6+zthickness)):
                    reinit = True
                    keepSwimming = True
                    mf_dirichlet[f] = 0
                    domain_id[f] = 12
                elif xDef[2] >= zRoof-zthickness and mf_dirichlet[f] != 6:
                    domain_id[f] = 4
                    reinit = True
                    mf_dirichlet[f] = 6
                    keepSwimming = True
        

        if reinit: # then solver needs to be updated
            bc_nanopillar = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
            bc_nanopillar_inner = DirichletBC(V1.sub(2), zDisplInner, mf_dirichlet, 5)
            bc_upper = DirichletBC(V1.sub(2), zDisplUpper, mf_dirichlet, 4)
            bc_upper_inner = DirichletBC(V1.sub(2), zDisplUpperInner, mf_dirichlet, 6)
            if not np.isinf(zRoof):
                bcs = [bc_nanopillar, bc_nanopillar_inner, bc_upper, 
                    bc_upper_inner, bc_symm1_y, bc_symm2_x]
            else:
                bcs = [bc_nanopillar, bc_nanopillar_inner, bc_symm1_y, bc_symm2_x]
            if custom_solver:
                problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
            else:
                solver = init_solver(Fvar, fmixed, bcs, Jvar)
        
        if keepSwimming:
            print(f"Computing idx = {idx}, outer iteration {it}, def = {uEval(0,0,2*nucRad2)[2]}")
        else:
            print(f"Done computing idx = {idx} after {it} iterations, def = {uEval(0,0,2*nucRad2)[2]}")

    # set next force values
    if pRamp[-1] >= pMax and not pumpedUp:
        pumpedUp = True
        # topFcn = project(dot(normals,Ttensor), V_vector)
        # topFcnInner = project(dot(inner_normals,Ttensor), V_vector)
        # topContactForce.assign(topFcn.copy())
        # topContactForce_inner.assign(topFcnInner.copy())
        # bc_nanopillar = DirichletBC(V1.sub(2), zDisplOuter, mf_dirichlet, 1)
        # bc_nanopillar_inner = DirichletBC(V1.sub(2), zDisplInner, mf_dirichlet, 5)
        # bcs = [bc_nanopillar, bc_nanopillar_inner, bc_symm1_y, bc_symm2_x]
        # if custom_solver:
        #     problem, solver = init_custom_solver(Fvar, fmixed, u, p, bcs)
        # else:
        #     solver = init_solver(Fvar, fmixed, bcs, Jvar)
        kRamp.append(kMin)
        pRamp.append(pMax)
    elif pumpedUp:
        # zRoof = 2*nucRad2 + u(0,0,2*nucRad2)[2]
        if kRamp[-1] < kMax:
            kRamp.append(min([kRamp[-1]+kInc, kMax]))
            pRamp.append(pMax)
        else:
            break # then done with this simulation
    else:
        pRamp.append(min([pRamp[-1]+dpress, pMax]))
        kRamp.append(kMin)   
    u_file.write(fmixed.sub(0), pRamp[-2]+kRamp[-2])
    if len(fmixed.sub(0).vector()) == len(u_prev.vector()):
        u_prev.vector()[:] = fmixed.sub(0).vector()[:]
        u_prev.vector().apply("insert")
    else:
        raise ValueError("Could not reassign u_prev!!")

    # zNP.append((zNP[-1]+u(0,0,0)[2])/2)
    # kRepel += 0.01

    # compute stretch for current configuration
    a_vector_new = J * dot(normals, inv(F))
    a_vector_new = project(a_vector_new, V_vector)
    a_vector.assign(a_vector_new)
    a_file.write(a_vector, pRamp[-1]+kRamp[-1])

    # save relevant vol and SAs
    inner_SA.append(assemble(a_scalar*ds_integrate(12))/inner_SA_ref)
    outer_SA.append(assemble(a_scalar*ds_integrate(10))/outer_SA_ref)
    vol.append(assemble(J*x[0]*dx)/vol_ref)

    np.savetxt(f"{results_folder}/inner_SA.txt", inner_SA)
    np.savetxt(f"{results_folder}/outer_SA.txt", outer_SA)
    np.savetxt(f"{results_folder}/vol.txt", vol)
