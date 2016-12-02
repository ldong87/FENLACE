from dolfin import *
import numpy as np

mesh = Mesh("xmlmesh.xml")

V = VectorFunctionSpace(mesh, "CG", 1) # displacement
g = Function(V)

# write g in serial
g_np = np.loadtxt("measuredDisp_1col.txt")   # Order: 1st node: x, y; 2nd node: x, y ...

dofs = dof_to_vertex_map(V)

g_np = g_np[dofs]

g.vector().set_local( g_np )
fid = HDF5File(mpi_comm_world(), "g.h5","w")
fid.write(g,"g")
fid.close()

fid = HDF5File(mpi_comm_world(), "mesh.h5","w")
fid.write(mesh, "mesh")
fid.close()


