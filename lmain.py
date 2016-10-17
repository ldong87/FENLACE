from dolfin import *
from dolfin_adjoint import *
#import pyipopt
import numpy as np

parameters["reorder_dofs_serial"] = False


mesh = Mesh("xmlmesh.xml")

V = VectorFunctionSpace(mesh, "CG", 1) # displacement
M = FunctionSpace(mesh, "CG", 1)       # mu, control space
v = TestFunction(V)
u = Function(V, name="State")
mu= interpolate(Constant(1), M, name="Control")
lam = Constant(2.0)

g_np = np.loadtxt("measuredDisp_1col.txt")
g = Function(V)
g.vector().set_local( g_np )

'''
# impose boundary conditions point by point
F = inner(mu/2 * (grad(v)+grad(v).T), grad(u)+grad(u).T)*dx + lam*inner(div(v),div(u))*dx 
class PinPoint(SubDomain):
    def __init__(self,p):
	self.p = p
	SubDomain.__init__(self)
    def inside(self, x, on_boundary):
	return np.linalg.norm(x-self.p)<1e-8
bc_coord = np.loadtxt("bc_coord.txt");
bc_disp = np.loadtxt("bc_disp.txt");
bc = []
for i in xrange(0, len(bc_coord)):
    if bc_disp[i,0] == 1:
        bc_0 = DirichletBC(V.sub(0), bc_disp[i,1], PinPoint([bc_coord[i,0],bc_coord[i,1]]), "pointwise")
	bc.append(bc_0)
    if bc_disp[i,2] == 1:
        bc_1 = DirichletBC(V.sub(1), bc_disp[i,3], PinPoint([bc_coord[i,0],bc_coord[i,1]]), "pointwise")
	bc.append(bc_1)

solve(F == 0, u,  bc)
'''



def forward(mu):
  class PinPoint(SubDomain):
    def __init__(self,p):
      self.p = p
      SubDomain.__init__(self)
    def inside(self, x, on_boundary):
      return np.linalg.norm(x-self.p) < DOLFIN_EPS
  class TopBot(SubDomain):
    def inside(self, x, on_boundary):
      return near(x[1], 0.) or near(x[1], 0.99)

  v = TestFunction(V)
  u = Function(V, name="State")
  F = inner(mu/2 * (grad(v)+grad(v).T), grad(u+g)+grad(u+g).T)*dx + lam*inner(div(v),div(u+g))*dx 
  bc = []
  bc_0 = DirichletBC(V.sub(0), Constant(0.0), "on_boundary")
  bc.append(bc_0)
  bc_1 = DirichletBC(V.sub(1), Constant(0.0), "on_boundary")
  bc.append(bc_1)
  solve(F == 0, u,  bc)
  return u
u = forward(mu)



#File("output/g.pvd") << g

#u_1 = project(u, V)
#File("output/u_1.pvd") << u_1

#u_sol_1 = project(u+g, V)
#File("output/u_sol_1.pvd") << u_sol_1

# create measured disp
'''
x = SpatialCoordinate(mesh)
#u_m_t = np.zeros((mesh.num_vertices(),2))
u_m_np = mesh.coordinates()
#np.savetxt("u_m_t",u_m_t)
'''
'''
u_m_np = np.loadtxt("measuredDisp.txt")
u_m = Function(V)
u_m.vector().set_local( np.append(u_m_np[:,0], u_m_np[:,1]) )
'''
u_m = g

#T = as_matrix( ((0, 0), (0, 1)) )
alpha = Constant(1e-5)

output = File("output/output_iterations_final.pvd")
mu_viz = Function(M, name="ControlVisualization")
def eval_cb(j, mu):
    mu_viz.assign(mu)
    output << mu_viz

#J = Functional((0.5*inner((u-u_m),u-u_m))*dx + 0.5*alpha*inner(grad(mu),grad(mu))*dx)
J = Functional((0.5*inner((u+g-u_m),u+g-u_m))*dx + 0.5*alpha*inner(grad(mu),grad(mu))*dx)
mu_c = Control(mu)
J_rf = ReducedFunctional(J, mu_c, eval_cb_post=eval_cb)


# scipy optimizer
mu_opt = minimize(J_rf, bounds=(1.0, 20.0), options={"gtol":1e-12}) 


'''
# TAO optimizer
problem = MinimizationProblem(J_rf)
parameters = { "monitor": None,
               "type": "blmvm",
               "bounds":(1.0, 20.0),
               "max_it": 300,
               #"subset_type": "matrixfree",
               "fatol": 1e-17,
               "frtol": 1e-17,
               "gatol": 1e-17,
               "grtol": 1e-17
             }
solver = TAOSolver(problem, parameters=parameters)
mu_opt = solver.solve()
'''

'''
ipopt optimizer
problem = MinimizationProblem(J_rf, bounds=(1.0,20.0))
parameters = {'maximum_iterations': 300}
solver = IPOPTSolver(problem, parameters=parameters)
mu_opt = solver.solve()
'''

u_sol = project(u+g, V)
File("output/u_sol.pvd") << u_sol
File("output/u_m.pvd") << u_m

plot(mu_opt, interactive=True, title="mu_opt")
#plot(u, interactive=True, title="u")
plot(u+g, interactive=True, title="u+g")
plot(u_m, interactive=True, title="u_m")

