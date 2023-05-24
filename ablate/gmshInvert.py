import os
import math
import gmsh
import argparse


def load_stp_and_mesh(filename, num_elements):
    gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)  # standard
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

    gmsh.model.add("stp")
    gmsh.merge(filename)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(), num_elements)
    gmsh.model.mesh.setRecombine(3, -1)  # Recombine for all volumes
    gmsh.model.mesh.generate(3)

    gmsh.write(os.path.splitext(filename)[0] + '.msh')

    gmsh.finalize()


def invert_mesh(filename, chunk_size):
    gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity", 0)

    gmsh.model.add("mesh")
    gmsh.merge(filename)

    all_entities = gmsh.model.getEntities()
    for dim, tag in all_entities:
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
        for element_type, elements in zip(element_types, element_tags):
            node_tags, local_coord, _ = gmsh.model.mesh.getNodesByElementType(element_type, tag=tag)
            local_coord_chunked = [local_coord[i:i + 3 * chunk_size] for i in
                                   range(0, len(local_coord), 3 * chunk_size)]

            for local_coord_chunk in local_coord_chunked:
                jacobians, determinants, _ = gmsh.model.mesh.getJacobians(element_type, local_coord_chunk, tag=tag)
                for i in range(len(jacobians)):
                    if jacobians[i] < 0:
                        gmsh.model.mesh.reverse([(dim, tag)])

    gmsh.write(filename)

    gmsh.finalize()


def spherical_shell():
    gmsh.initialize()

    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

    gmsh.model.add('spherical_shell')

    r1 = 1.0
    r2 = 2.0

    outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, r2)
    inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, r1)

    gmsh.model.occ.synchronize()

    # Cut the inner sphere from the outer sphere to create the shell
    shell_volume, _ = gmsh.model.occ.cut([(3, outer_sphere)], [(3, inner_sphere)])

    gmsh.model.occ.synchronize()

    lc = 0.1
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    outer_surface = gmsh.model.addPhysicalGroup(2, [outer_sphere])
    gmsh.model.setPhysicalName(2, outer_surface, "R2")

    inner_surface = gmsh.model.addPhysicalGroup(2, [inner_sphere])
    gmsh.model.setPhysicalName(2, inner_surface, "R1")

    gmsh.model.mesh.generate(3)

    gmsh.write('spherical_shell.msh')

    gmsh.finalize()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Invert negative Jacobian elements in a mesh file.')
    parser.add_argument('--file', type=str, required=True, help='Mesh file path (.msh)')
    parser.add_argument('--chunk_size', type=int, required=True, help='Chunk size for processing determinants')
    args = parser.parse_args()

    # Initialize the Gmsh Python API
    gmsh.initialize()

    # Create a new model
    gmsh.model.add("new_model")

    # Add two spheres to the model
    outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, 10)
    inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, 5)

    # Synchronize the data from OCC CAD kernel to Gmsh model
    gmsh.model.occ.synchronize()

    # Get the boundaries (surfaces) of the spheres and extract the tags
    outer_surface = [tag for dim, tag in gmsh.model.getBoundary([(3, outer_sphere)])]
    inner_surface = [tag for dim, tag in gmsh.model.getBoundary([(3, inner_sphere)])]

    # Perform the boolean difference operation
    diff_volume, _ = gmsh.model.occ.cut([(3, outer_sphere)], [(3, inner_sphere)])

    # Synchronize the data from OCC CAD kernel to Gmsh model
    gmsh.model.occ.synchronize()

    # Label the outer and inner surfaces as physical groups
    gmsh.model.addPhysicalGroup(2, outer_surface, tag=1)
    gmsh.model.setPhysicalName(2, 1, "R2")
    gmsh.model.addPhysicalGroup(2, inner_surface, tag=2)
    gmsh.model.setPhysicalName(2, 2, "R1")

    # Add physical groups for the volumes
    gmsh.model.addPhysicalGroup(3, [diff_volume[0][1]], tag=3)
    gmsh.model.setPhysicalName(3, 2, "volume")

    # Set the characteristic length (mesh size)
    gmsh.model.mesh.setSize(gmsh.model.getBoundary([(3, diff_volume[0][1])], recursive=True), 3.5)

    # Generate the 2D mesh first
    gmsh.model.mesh.generate(2)

    # Recombine the mesh
    gmsh.model.mesh.recombine()

    # Set the subdivision algorithm
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

    # Refine the mesh
    gmsh.model.mesh.refine()

    # If you still want a 3D mesh, you can generate the 3D mesh after the refinement:
    gmsh.model.mesh.generate(3)

    # Optional: Save the mesh to a file
    gmsh.write("spherical_shell.msh")

    # Finalize the Gmsh Python API
    gmsh.finalize()



