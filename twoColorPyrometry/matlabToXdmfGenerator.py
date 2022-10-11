import argparse
import pathlib
import xml.etree.ElementTree as ET
import h5py

from support.supportPaths import expand_path


def write_data_item(xdmf_parent, hdf5_item, hdf5_filename):
    dataItem = ET.SubElement(xdmf_parent, 'DataItem')
    dataItem.set('Dimensions', ' '.join(map(str, hdf5_item.shape)))
    dataItem.set('NumberType', 'Float')
    dataItem.set('Precision', '8')
    dataItem.set('Format', 'HDF')
    dataItem.text = hdf5_filename + ":" + hdf5_item.name


# store some dimensional information
topology_type = {
    2: "2DCoRectMesh",
    3: "3DCoRectMesh"
}

geometry_type = {
    2: "Origin_DxDy",
    3: "Origin_DxDyDz"
}


# generate the xdmf file xml for this file
def generate_xdmf(hdf5_file, xdmf_file, root='main'):
    # Load in the hdf5 file
    hdf5 = h5py.File(hdf5_file, 'r')

    # Check each of the cell fields
    hdf5_fields = hdf5[root]

    # store the dimensions, note this will be z, y, x
    gridDim = []

    # create a list of items to output
    fieldNames = []
    for field_name in hdf5_fields.keys():
        # if field data
        if hdf5_fields[field_name].ndim > 1 and hdf5_fields[field_name].shape[1] > 1:
            fieldNames.append(field_name)
            # check the dim
            if gridDim:
                if gridDim != hdf5_fields[field_name].shape:
                    raise Exception("All fields must be of same size")
            else:
                gridDim = hdf5_fields[field_name].shape

    # convert the gridDim to a string rep (note add one for cell/face info)
    grid_cell_dim_string = ' '.join(map(lambda d: str(d + 1), gridDim))

    # Create the root xdmf file element
    xdmf = ET.Element('Xdmf')
    domain = ET.SubElement(xdmf, 'Domain')
    grid = ET.SubElement(domain, 'Grid')
    grid.set('Name', 'mesh')
    grid.set('GridType', 'Uniform')

    # determine the dim of the grid
    grid_dim = len(gridDim)

    # Set the baseline topology
    topology = ET.SubElement(grid, 'Topology')
    topology.set('TopologyType', topology_type[grid_dim])
    topology.set('NumberOfElements', grid_cell_dim_string)

    # add the simple geometry describing a structured grid
    geometry = ET.SubElement(grid, 'Geometry')
    geometry.set('GeometryType', geometry_type[grid_dim])
    write_data_item(geometry, hdf5_fields['start'], hdf5_file.name)
    write_data_item(geometry, hdf5_fields['discretization'], hdf5_file.name)

    # Add in the field data
    for field_name in fieldNames:
        fieldData = ET.SubElement(grid, 'Attribute')
        fieldData.set('Name', field_name)
        fieldData.set('AttributeType', 'Scalar')
        fieldData.set('Center', 'Cell')
        write_data_item(fieldData, hdf5_fields[field_name], hdf5_file.name)

    # write to file
    tree = ET.ElementTree(xdmf)
    ET.indent(tree, space="\t", level=0)
    with open(xdmf_file, 'w') as f:
        f.write('<?xml version="1.0" ?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        tree.write(f, xml_declaration=False, encoding='unicode')


# Main function to parser input files and run the document generator
def parse():
    parser = argparse.ArgumentParser(description='Generate an xdmf file from MatLab data files holding structed data. '
                                                 'See https://github.com/cet-lab/experimental-post-processing/wiki'
                                                 '/Matlab-To-XdmfGenerator for details.  ')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The path to the hdf5 file(s) containing the structured data.  A wild card can be used '
                             'to supply more than one file.')
    args = parser.parse_args()

    # expand any wild cards in the path name
    hdf5_paths = expand_path(args.hdf5_file)

    # convert with path
    for hdf5_file in hdf5_paths:
        # based upon the input file get the output file
        xdmf_file = hdf5_file.parent / (hdf5_file.stem + ".xdmf")

        # create component markdown
        generate_xdmf(hdf5_file, xdmf_file)


# parse based upon the supplied inputs
if __name__ == "__main__":
    parse()
