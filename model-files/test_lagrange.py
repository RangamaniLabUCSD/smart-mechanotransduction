import dolfin as d
from smart import mesh_tools
import numpy as np
import pathlib, sys
here = pathlib.Path(__file__).parent

loaded= mesh_tools.load_mesh("/root/shared/gitrepos/smart-mechanotransduction"
                                      "/meshes/nanopillars_movenuc/nanopillars_movenuc-1.6"
                                      "/spreadCell_mesh.h5")
mesh = loaded.mesh
mf2 = loaded.mf_facet
mf3 = loaded.mf_cell
cyto_mesh = d.create_meshview(mf3, 1)
nuc_mesh = d.create_meshview(mf3, 2)
pm_mesh = d.create_meshview(mf2, 10)
ne_mesh = d.create_meshview(mf2, 12)

V = d.VectorFunctionSpace(mesh, "P", 1)
Vscalar = d.FunctionSpace(mesh, "P", 1)
u = d.Function(V)
du = d.TrialFunction(V)
v = d.TestFunction(V)

# test translation of nucleus
# Kinematics
dim = 3
I = d.Identity(dim)             # Identity tensor

# global kinematics
F = I + d.grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor
J  = d.det(F)
I1 = d.tr(C) * J**(-2/3)
I2 = 0.5*(I1**2 - d.tr(C*C)) * J**(-4/3)

# Elasticity parameters
E_cyto = 1000.0
E_nuc = 5000.0
bulk_mod = 1e4

# Stored strain energy density (neo-Hookean)
psi_cyto = E_cyto*(I1-dim) + bulk_mod*(J-1)**2
psi_nuc = E_nuc*(I1-dim) + bulk_mod*(J-1)**2

# markers for symmetry conditions
mf_symm = d.MeshFunction("size_t", mesh, 2, 0)
class SymmAxis1(d.SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.001 and on_boundary
class SymmAxis2(d.SubDomain):
    def inside(self, x, on_boundary):
        return (np.arctan2(x[1],x[0]) > 0.99*np.pi/4 or 
                (x[1]==0 and x[0]==0)) and on_boundary
symm1 = SymmAxis1()
symm2 = SymmAxis2()
symm1.mark(mf_symm, 1)
symm2.mark(mf_symm, 2)
ds_global = d.Measure("ds", domain=mesh, subdomain_data=mf_symm)

# Dirichlet BC of zero deformation at PM
bc_PM = d.DirichletBC(V, d.Constant((0.0,0.0,0.0)), mf2, 10)
bc_NE = d.DirichletBC(V, d.Constant((0.0,0.0,-0.1)), mf2, 12)
# bc_symm1 = d.DirichletBC(V.sub(1), d.Constant(0.0), mf_symm, 1)
bcs = [bc_PM, bc_NE]
dx_global = d.Measure("dx", domain=mesh, subdomain_data=mf3)

# Total potential energy
Pi_cyto = psi_cyto*dx_global(1)
Pi_nuc = psi_nuc*dx_global(2)

Pi_global = Pi_cyto + Pi_nuc

# Compute first variation of Pi (directional derivative about u in the direction of v)
# F = d.derivative(Pi_cyto, fmixed, v_cyto) + d.derivative(Pi_nuc, fmixed, v_nuc)
Fvar = d.derivative(Pi_global, u, v)

# apply penalties for violations of symmetry
penVal = 1e8
Fvar = Fvar + penVal*(u[1])*v[1]*ds_global(1) + penVal*(u[0]-u[1])*(v[0]-v[1])*ds_global(2)

# Compute Jacobian of F
# J = d.derivative(F, u_cyto, du_cyto) + d.derivative(F, u_nuc, du_cyto)
Jvar = d.derivative(Fvar, u, du)

# Solve variational problem
problem = d.NonlinearVariationalProblem(Fvar, u, bcs, J=Jvar)
solver = d.NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["newton_solver"]["absolute_tolerance"] = 1E-8
prm["newton_solver"]["relative_tolerance"] = 1E-6
prm["newton_solver"]["maximum_iterations"] = 100
prm["newton_solver"]["linear_solver"] = 'mumps'

idx = 0
u_file = d.XDMFFile(f"test_lagrange/u_lagrange.xdmf")
u_file.parameters["flush_output"] = True
u_file.write(u, idx)
J_file = d.XDMFFile(f"test_lagrange/J_lagrange.xdmf")
J_file.parameters["flush_output"] = True
Jproj = d.project(J, Vscalar)
J_file.write(Jproj, idx)

ref_vol = d.assemble(1.0*dx_global)
vol_vec = [d.assemble(Jproj*dx_global)]
assert ref_vol == vol_vec[0]
print(f'Starting volume is {vol_vec[0]}')

while idx < 16:
    idx += 1
    solver.solve()
    print(f"Done with idx = {idx}")
    u_file.write(u, idx)
    Jproj.assign(d.project(J, Vscalar))
    if np.any(Jproj.vector()[:] < 0.0):
        Jproj.vector()[Jproj.vector()[:] < 0.0] = 0.0
    J_file.write(Jproj, idx)
    bc_NE.set_value(d.Constant((0.0,0.0,-0.1*(idx+1))))
    vol_vec.append(d.assemble(Jproj*dx_global))
    np.savetxt('test_lagrange/vol_vec.txt', vol_vec)
    print(f'Idx={idx}, volume is {vol_vec[-1]}')