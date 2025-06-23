from dolfin import *
from smart import mesh_tools
import numpy as np

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

# Create mesh and define function space
nucRad1 = 6.625#5.5
nucRad2 = 3.0
thickness = 0.05
mesh_ref, mf1, mf2 = mesh_tools.create_2Dcell_xy(outerExpr=f"(r/{nucRad1})**2 + ((z-{nucRad2})/{nucRad2})**2 - 1",
                                                 innerExpr=f"(r/{nucRad1-thickness})**2 + ((z-{nucRad2})/{nucRad2-thickness})**2 - 1", 
                                                 hEdge=0.02, hInnerEdge=0.02)
# mesh_ref, mf1, mf2 = mesh_tools.create_ellipses(xrad_outer=nucRad1, yrad_outer=nucRad2,
#                                                 xrad_inner=nucRad1-thickness, yrad_inner=nucRad2-thickness, 
#                                                hEdge=0.1, hInnerEdge=0.1)
# for c in cells(mesh_ref):
#     if c.midpoint().z() > -0.8*nucRad:
#         mf3[c] = 0
mesh = create_meshview(mf2, 1)
mf_surf = MeshFunction("size_t", mesh, 1, 0)
mf_dirichlet = MeshFunction("size_t", mesh, 1, 0)
class OuterSurf(SubDomain):
    def inside(self, x, on_boundary):
        return pow(pow(x[0]/nucRad1,2) + pow((x[1]-nucRad2)/nucRad2,2),0.5) > 0.999 and on_boundary
class InnerSurf(SubDomain):
    def inside(self, x, on_boundary):
        return pow(pow(x[0]/(nucRad1-thickness),2) + 
                   pow((x[1]-nucRad2)/(nucRad2-thickness),2),0.5) < 1.001 and on_boundary
class NanopillarContact(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < 0.02*nucRad1 and x[1] < nucRad2 and on_boundary
                and pow(pow(x[0]/nucRad1,2) + pow((x[1]-nucRad2)/nucRad2,2),0.5) > 0.999)
class SymmAxis(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary
outerSurf = OuterSurf()
innerSurf = InnerSurf()
outerSurf.mark(mf_surf, 10) # mark outer surface points as 10
innerSurf.mark(mf_surf, 12) # mark inner surface points as 12

nanopillar = NanopillarContact()
symm = SymmAxis()
nanopillar.mark(mf_dirichlet, 1) # mark nanopillar contact with 1
symm.mark(mf_dirichlet, 2)

V_vector = FunctionSpace(mesh, VectorElement("P", mesh.ufl_cell(), degree = 1, dim = 3))
# V_tensor = FunctionSpace(mesh_bound, TensorElement("P", mesh_bound.ufl_cell(), degree = 1, dim = 3))
# V_scalar = FunctionSpace(mesh_bound, "P", 1)
normal_expr = Expression(("(x[0]/pow(a,2))/sqrt(pow(x[0],2)/pow(a,4) + pow(x[1]-b,2)/pow(b,4))", "0.0",
                            "((x[1]-b)/pow(b,2))/sqrt(pow(x[0],2)/pow(a,4) + pow(x[1]-b,2)/pow(b,4))"),
                            a=nucRad1, b=nucRad2, degree=1)
normals = project(normal_expr, V_vector)

# mesh = RectangleMesh(Point(0.0, 0.0), Point(1.5*nucRad1, thickness), 151, 5)
npSpacing = 2.0
npRad = 0.5
xMax = 2.0 #np.floor((nucRad/2 - 2*npRad) / npSpacing) * npSpacing
xNP = np.arange(-xMax, xMax+1e-12, npSpacing)
yNP = np.arange(-xMax, xMax+1e-12, npSpacing)
xNP, yNP = np.meshgrid(xNP, yNP)
xNP = xNP.flatten()
yNP = yNP.flatten()

x = SpatialCoordinate(mesh)
def axisymm_grad(v): # axis of symm is x[0]=0
    # in addition to this definition, x[0] is included in all integrals
    return sym(as_tensor([[v[0].dx(0), 0, v[0].dx(1)],
                        [0, v[0]/x[0], 0],
                        [v[1].dx(0), 0, v[1].dx(1)]]))

File("nuc_indent/test_shell.pvd") << mesh

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
c = Expression(("0.0", "0.0"), degree=1)
zDispl = Expression("-r2*(1-sqrt(1-pow(x[0]/r1,2)))", degree=1, r1=nucRad1, r2=nucRad2)
bc_nanopillar = DirichletBC(V1.sub(1), zDispl, mf_dirichlet, 1)
# bc_fixed2 = DirichletBC(V1.sub(1), zDispl, fixed_bound2)
bc_symm = DirichletBC(V1.sub(0), Constant(0.0), mf_dirichlet, 2)
bcs = [bc_nanopillar, bc_symm]

# Define functions
# du = TrialFunction(V1)
# v = TestFunction(V1)
# u = Function(V1)
(du, dp) = TrialFunctions(Vmixed)            # Incremental displacement and dp
(v, q)  = TestFunctions(Vmixed)             # Test function
fmixed  = Function(Vmixed)                 # Displacement from previous iteration
(u, p) = fmixed.split()
# (utest,ptest) = fmixed.split(deepcopy=True)
B  = Constant((0.0, 0.0))  # Body force per unit volume
# T  = Constant((-0.001,  0.0, 0.0))  # Traction force on the boundary

domain_id = MeshFunction("size_t", mesh, 1, 0)
for f in facets(mesh):
    for i in range(len(xNP)):
        # xCur = xNP[i]
        # yCur = yNP[i]
        # rCur = np.sqrt((f.midpoint().x()-xCur)**2 + (f.midpoint().y()-yCur)**2)
        # RCur = np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2 + f.midpoint().z()**2)
        testEdge = (f.midpoint().x()/nucRad1)**2 + ((f.midpoint().y()-nucRad2)/nucRad2)**2
        # if rCur <= npRad and f.midpoint().z() < 1.0 and np.isclose(testEdge,1): #np.isclose(f.midpoint().z(),0.0)  and rCur <= npRad:
        #     domain_id[f] = 1
        if f.midpoint().y() > 1.2*nucRad2 and (f.midpoint().x() > 0.0):# and f.midpoint().x() < 0.05*nucRad1):# or (
            # (f.midpoint().x() > 0.5*nucRad1 and f.midpoint().x() < 0.55*nucRad1)):#(testEdge>0.99): #np.isclose(f.midpoint().z(),0.0)  and rCur <= npRad:
            domain_id[f] = 1
        # elif f.midpoint().y() > 0.99*thickness:
        #     domain_id[f] = 2
ds = Measure('ds', domain=mesh, subdomain_data=domain_id)

# Kinematics
d = 3#u.geometric_dimension()
I = variable(Identity(d))             # Identity tensor
F = variable(I + axisymm_grad(u))             # Deformation gradient
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
results_folder = "nuc_indent_1e5J_NPForce_mixed_incrBound"
u_file = XDMFFile(f"{results_folder}/u_np_ellipsoid.xdmf")
u_file.parameters["flush_output"] = True
u_file.write(u, idx)
a_vector = J * dot(normals, inv(F))
a_vector = project(a_vector, V_vector)
a_file = XDMFFile(f"{results_folder}/a_np_ellipsoid.xdmf")
a_file.parameters["flush_output"] = True
a_file.write(a_vector, idx)
kMin = 0.0
kRepel = 1e6
kRamp = [kMin]

u.set_allow_extrapolation(True)

zIndent = 4.0

a_scalar = ufl.sqrt(a_vector[0]**2 + a_vector[1]**2 + a_vector[2]**2)
ds_integrate = Measure('ds', domain=mesh, subdomain_data=mf_surf)
inner_SA = []
outer_SA = []
vol = []
inner_SA_ref = assemble(1.0*x[0]*ds_integrate(12))
outer_SA_ref = assemble(1.0*x[0]*ds_integrate(10))
vol_ref = assemble(1.0*x[0]*dx)

while u(0,2*nucRad2)[1] > -zIndent:#zNP[-1] < zFinal-1e-6:

    curForce = kRamp[-1]
    print(f"Current force is {curForce}")
    T = Expression(("0.0", "curForce"), degree=1, curForce=-curForce)
    # Ptop = Expression(("0.0", "pressure"), degree=1, pressure=1e-6)

    # Convert potential energy to first Piola-Kirchoff stress tensor
    Ttensor = diff(psi, F)
    # Pi = ufl.diff(psi, F) + p * J * ufl.inv(F.T)
    # Pi = psi*x[0]*dx - dot(T,u)*x[0]*ds(1) #+ dot(Ptop,u)*x[0]*ds(2)
    # Fvar = derivative(Pi, u, v) #derivative(Pi, u, v) + derivative(Pi, p, q)
    # Fvar = derivative(Pi, u, v) - q*(J - 1)*x[0]*dx
    Fvar = inner(axisymm_grad(v), Ttensor)*x[0]*dx - inner(v, T)*x[0]*ds(1) + inner(q, 1e5*(J-1) + p) * x[0] * dx
    # Compute Jacobian of F
    Jvar = derivative(Fvar, u, du) + derivative(Fvar, p, dp)
    # Define problem and solver with custom settings
    problem = NonlinearVariationalProblem(Fvar, fmixed, bcs, J=Jvar)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-8
    prm["newton_solver"]["relative_tolerance"] = 1E-6
    prm["newton_solver"]["maximum_iterations"] = 100
    # try solving, if diverges, take smaller step
    try:
        solver.solve()
        if kRamp[-1] < kRepel:
            kRamp.append(min([kRamp[-1]+0.001, kRepel]))
            # continue
        elif kRamp[-1] == kRepel:
            kRamp = [kMin]
        else:
            raise ValueError("k cannot be greater than kRepel")
    except:
        print(f"Resetting kRamp from {kRamp[-1]} to")
        if len(kRamp) == 1:
            kRamp[-1] = kRamp[-1]/2
        else:
            kRamp[-1] = (kRamp[-1]+kRamp[-2])/2
        print(f"{kRamp[-1]} because solve failed")
        continue

    idx += 1
    print(f"Done computing idx = {idx}, def = {u(0,2*nucRad2)[1]}")
    u_file.write(u, idx)
    # update bcs as needed
    coords = mf_dirichlet.mesh().coordinates()
    for f in facets(mesh):
        xCur = [f.midpoint().x(), f.midpoint().y()]
        xDef = xCur + u(xCur)
        if mf_surf[f] == 10:
            if xDef[1] <= 0 and ((xDef[0] <= 0.1*nucRad1) or (xDef[0] <= 0.6*nucRad1 and xDef[0] > 0.5*nucRad1)):
                mf_dirichlet[f] = 1
    bc_nanopillar = DirichletBC(V1.sub(1), zDispl, mf_dirichlet, 1)
    bcs = [bc_nanopillar, bc_symm]
    
    # zNP.append((zNP[-1]+u(0,0,0)[2])/2)
    # kRepel += 0.01

    # compute stretch for current configuration
    a_vector_new = J * dot(normals, inv(F))
    a_vector_new = project(a_vector_new, V_vector)
    a_vector.assign(a_vector_new)
    a_file.write(a_vector, idx)

    # save relevant vol and SAs
    inner_SA.append(assemble(a_scalar*x[0]*ds_integrate(12))/inner_SA_ref)
    outer_SA.append(assemble(a_scalar*x[0]*ds_integrate(10))/outer_SA_ref)
    vol.append(assemble(J*x[0]*dx)/vol_ref)

    np.savetxt(f"{results_folder}/inner_SA.txt", inner_SA)
    np.savetxt(f"{results_folder}/outer_SA.txt", outer_SA)
    np.savetxt(f"{results_folder}/vol.txt", vol)
