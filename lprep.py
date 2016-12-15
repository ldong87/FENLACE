from dolfin import *
import numpy as np

mesh = Mesh("xmlmesh.xml")

V = VectorFunctionSpace(mesh, "CG", 1) # displacement, state space
uMeas = Function(V)

# write g in serial
uMeas_np = np.loadtxt("measuredDisp_1col.txt")   # Order: 1st node: x, y; 2nd node: x, y ...

dofs = dof_to_vertex_map(V)

uMeas_np = uMeas_np[dofs]

uMeas.vector().set_local( uMeas_np )
fid = HDF5File(mpi_comm_world(), "mesh_measDisp.h5","w")
fid.write(uMeas,"uMeas")
fid.write(mesh, "mesh")
fid.close()
