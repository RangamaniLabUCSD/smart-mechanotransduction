from dolfin import *
from smart import mesh_tools
import numpy as np
import pathlib
import sys
import argparse, logging

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

parser = argparse.ArgumentParser()
add_nucmech_arguments(parser)
args = vars(parser.parse_args())
run_local = False
if run_local:
    cur_dir = str(pathlib.Path.cwd() / "..")
    args["mesh_folder"] = pathlib.Path("")
    args["max_force"] = 0.01
    args["start_force"] = 0.0
    args["u0"] = pathlib.Path("")
    args["bulk_mod"] = 1e6
    args["nanopillar_radius"] = 0.25
    args["nanopillar_height"] = 1.0
    args["nanopillar_spacing"] = 2.5
    args["outdir"] = pathlib.Path(f"/root/shared/gitrepos/nuc_indent_1e6J_npRad{args['nanopillar_radius']}"
                                  f"_npSpacing{args['nanopillar_spacing']}")

# Create mesh and define function space
nucRad1 = 6.625#5.5
nucRad2 = 3.0
thickness = 0.05

mesh_ref, mf2, mf3 = mesh_gen.NE_mesh(rRad=nucRad1, zRad=nucRad2, thickness=thickness,
                                      hEdge=0.2, hInnerEdge=0.2, sym_fraction=0.25)        
mesh = create_meshview(mf3, 1)

# define nanopillar dimensions for current case
npSpacing = args["nanopillar_spacing"]
npRad = args["nanopillar_radius"]
xMax = np.ceil((max([nucRad1,nucRad2])) / npSpacing) * npSpacing
xNP = np.arange(-xMax, xMax+1e-12, npSpacing)
yNP = np.arange(-xMax, xMax+1e-12, npSpacing)
xNP, yNP = np.meshgrid(xNP, yNP)
xNP = xNP.flatten()
yNP = yNP.flatten()

mf_surf = MeshFunction("size_t", mesh, 2, 0)
mf_dirichlet = MeshFunction("size_t", mesh, 2, 0)
mf_edge = MeshFunction("size_t", mesh, 2, 0)
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
        return (np.sqrt(x[0]**2 + x[1]**2) < 0.8*npRad and x[2] < nucRad2 and on_boundary
                and pow(pow(x[0]/nucRad1,2) + pow(x[1]/nucRad1,2) + 
                        pow((x[2]-nucRad2)/nucRad2,2),0.5) > 0.999)
        # return on_boundary and near(x[2], 0.0) and (np.sqrt(x[0]**2 + x[1]**2) < 0.2)
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
symm1 = SymmAxis1()
symm2 = SymmAxis2()
nanopillar.mark(mf_dirichlet, 1) # mark nanopillar contact with 1
symm1.mark(mf_dirichlet, 2)
symm2.mark(mf_dirichlet, 3)

# symmEdge = SymmEdge()
# symmEdge.mark(mf_edge, 1)

V_vector = FunctionSpace(mesh, VectorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
# V_tensor = FunctionSpace(mesh, TensorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
# V_scalar = FunctionSpace(mesh, "P", 1)
normal_expr = Expression(("(x[0]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))", 
                          "(x[1]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))",
                          "((x[2]-b)/pow(b,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2]-b,2)/pow(b,4))"),
                            a=nucRad1, b=nucRad2, degree=1)
normals = project(normal_expr, V_vector)

# mesh = RectangleMesh(Point(0.0, 0.0), Point(1.5*nucRad1, thickness), 151, 5)

x = SpatialCoordinate(mesh)
# results_folder = "/root/scratch/nuc_indent_1e6J_eighth"
results_folder = args["outdir"]
File(f"{results_folder}/test_shell.pvd") << mesh

# Define mixed function space for displacement (u) and Lagrange multiplier (p)
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
# zDispl = Expression(("0.0","0.0","-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))"), degree=1, r1=nucRad1, r2=nucRad2)
zDispl = Expression("-r2*(1-sqrt(1-pow(x[0]/r1,2)-pow(x[1]/r1,2)))", degree=1, r1=nucRad1, r2=nucRad2)
# zDispl = Expression("0.0", degree=1)
bc_nanopillar = DirichletBC(V1.sub(2), zDispl, mf_dirichlet, 1)
# bc_fixed2 = DirichletBC(V1.sub(1), zDispl, fixed_bound2)
bc_symm1_x = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 2)
bc_symm1_y = DirichletBC(V1.sub(1), Constant(0.0), mf_dirichlet, 2)
bc_symm2_x = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 3)
bc_symm2_y = DirichletBC(V1.sub(1), Constant(0.0), mf_dirichlet, 3)
# bc_symmEdge1 = DirichletBC(V1.sub(0), Constant(0.0), mf_edge, 1)
# bc_symmEdge2 = DirichletBC(V1.sub(1), Constant(0.0), mf_edge, 1)
bcs = [bc_nanopillar, bc_symm1_y, bc_symm2_x]

# Define functions
# du = TrialFunction(V1)
# v = TestFunction(V1)
# u = Function(V1)
(du, dp) = TrialFunctions(Vmixed)            # Incremental displacement and dp
(v, q)  = TestFunctions(Vmixed)             # Test function
fmixed  = Function(Vmixed)                 # Displacement from previous iteration
(u, p) = fmixed.split()

domain_id = MeshFunction("size_t", mesh, 2, 0)
for f in facets(mesh):
    if mf_dirichlet[f] == 2:
        domain_id[f] = 2
    elif mf_dirichlet[f] == 3:
        domain_id[f] = 3
    elif f.midpoint().z() > 1.2*nucRad2 and mf_surf[f] == 10:
        domain_id[f] = 1
    # if (np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2) > 0.4 and 
    #     np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2) < 0.5):
    #         domain_id[f] = 1
ds = Measure('ds', domain=mesh, subdomain_data=domain_id)

# Kinematics
d = 3#u.geometric_dimension()
I = variable(Identity(d))             # Identity tensor
F = variable(I + grad(u))             # Deformation gradient
C = variable(F.T*F)                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
I1 = variable(tr(C))
I2 = variable(0.5*(I1**2 - tr(C*C)))
J  = variable(det(F))

# rescale I1, I2
I1 = I1 * J**(-2/3)
I2 = I2 * J**(-4/3)

# Elasticity parameters
E1, E2 = Constant(5000.0), Constant(1000.0)

# Stored strain energy density (incompressible Mooney Rivlin model)
# psi = (mu/2)*(I1 - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
psi = E1*(I1-3) + E2*(I2-3) - p*(J-1)
# psi = E1*(I1-3) + 4e4*(J-1)**2

zNP = [-0.05]#[-nucRad - 0.05]
idx = 0
zMove = 0.1
zStep = 0.1
zFinal = zNP[-1] + zMove
u_file = XDMFFile(f"{results_folder}/u_np_ellipsoid.xdmf")
u_file.parameters["flush_output"] = True
u_file.write(u, idx)
a_vector = J * dot(normals, inv(F))
a_vector = project(a_vector, V_vector)
a_file = XDMFFile(f"{results_folder}/a_np_ellipsoid.xdmf")
a_file.parameters["flush_output"] = True
a_file.write(a_vector, idx)
kMin = args["start_force"]
kMax = args["max_force"]
kRamp = [kMin]

u.set_allow_extrapolation(True)

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

while u(0,0,2*nucRad2)[2] > -zIndentMax:#zNP[-1] < zFinal-1e-6:

    curForce = kRamp[-1]
    print(f"Current force is {curForce}")
    T = Expression(("0.0", "0.0", "curForce"), degree=1, curForce=-curForce)
    # Ptop = Expression(("0.0", "pressure"), degree=1, pressure=1e-6)

    # Convert potential energy to first Piola-Kirchoff stress tensor
    Ttensor = diff(psi, F)
    # Pi = ufl.diff(psi, F) + p * J * ufl.inv(F.T)
    # Pi = psi*x[0]*dx - dot(T,u)*x[0]*ds(1) #+ dot(Ptop,u)*x[0]*ds(2)
    # Fvar = derivative(Pi, u, v) #derivative(Pi, u, v) + derivative(Pi, p, q)
    # Fvar = derivative(Pi, u, v) - q*(J - 1)*x[0]*dx
    Fvar = (inner(grad(v), Ttensor)*dx - inner(v, T)*ds(1) + 
            inner(q, args["bulk_mod"]*(J-1) + p) * dx) # + inner(q, u[0]-u[1])*ds(3) + inner(q, u[1])*ds(2)
    # Compute Jacobian of F
    Jvar = derivative(Fvar, u, du) + derivative(Fvar, p, dp)
    # Define problem and solver with custom settings
    problem = NonlinearVariationalProblem(Fvar, fmixed, bcs, J=Jvar)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-8
    prm["newton_solver"]["relative_tolerance"] = 1E-6
    prm["newton_solver"]["maximum_iterations"] = 100
    prm["newton_solver"]["linear_solver"] = 'mumps'
    # try solving, if diverges, take smaller step
    try:
        solver.solve()
        if kRamp[-1] < kMax:
            kRamp.append(min([kRamp[-1]+1e-3, kMax]))
        else:
            break # then done with this simulation
    except:
        print(f"Resetting kRamp from {kRamp[-1]} to")
        if len(kRamp) == 1:
            kRamp[-1] = kRamp[-1]/2
        else:
            kRamp[-1] = (kRamp[-1]+kRamp[-2])/2
        print(f"{kRamp[-1]} because solve failed")
        continue

    idx += 1
    print(f"Done computing idx = {idx}, def = {u(0,0,2*nucRad2)[2]}")
    u_file.write(u, idx)
    # update bcs as needed
    coords = mf_dirichlet.mesh().coordinates()
    for f in facets(mesh):
        if mf_surf[f] == 10:
            xCur = [f.midpoint().x(), f.midpoint().y(), f.midpoint().z()]
            uCur = u(xCur)
            xDef = xCur + uCur
            if xDef[2] <= 0 and mf_dirichlet[f] == 0: # not touched this nanopillar yet
                # rDef = np.sqrt(xDef[0]**2 + xDef[1]**2)
                all_dist = np.sqrt((xNP-xDef[0])**2 + (yNP-xDef[1])**2)
                if np.any(all_dist <= npRad):
                    mf_dirichlet[f] = 1
                    # bc_num = len(bcs) + 1
                    # mf_dirichlet[f] = bc_num
                    # bcs.append(DirichletBC(V1, Constant((uCur[0], uCur[1], -xCur[2])), mf_dirichlet, bc_num))
            # if xDef[2] <= 0 and ((rDef <= 0.1*nucRad1)):# or (xDef[0] <= 0.6*nucRad1 and xDef[0] > 0.5*nucRad1)):
            #     mf_dirichlet[f] = 1
    
    bc_nanopillar = DirichletBC(V1.sub(2), zDispl, mf_dirichlet, 1)
    bcs = [bc_nanopillar, bc_symm1_y, bc_symm2_x]
        
    # zNP.append((zNP[-1]+u(0,0,0)[2])/2)
    # kRepel += 0.01

    # compute stretch for current configuration
    a_vector_new = J * dot(normals, inv(F))
    a_vector_new = project(a_vector_new, V_vector)
    a_vector.assign(a_vector_new)
    a_file.write(a_vector, idx)

    # save relevant vol and SAs
    inner_SA.append(assemble(a_scalar*ds_integrate(12))/inner_SA_ref)
    outer_SA.append(assemble(a_scalar*ds_integrate(10))/outer_SA_ref)
    vol.append(assemble(J*x[0]*dx)/vol_ref)

    np.savetxt(f"{results_folder}/inner_SA.txt", inner_SA)
    np.savetxt(f"{results_folder}/outer_SA.txt", outer_SA)
    np.savetxt(f"{results_folder}/vol.txt", vol)
