import argparse
import pathlib

import cantera
import h5py
import yaml

from chrest.supportPaths import expand_path

# parse based upon the supplied inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes the mean and the rms')
    parser.add_argument('--file', dest='hdf5_file', type=pathlib.Path, required=True,
                        help='The input hdf5_file')
    parser.add_argument('--mechanism', dest='mechanism_file', type=pathlib.Path, required=True,
                        help='The mechanism file used to generate the hdf5_file')
    parser.add_argument('--error_repor_tol', dest='error_report_tolerance', type=float, required=False,
                        help='Any errors about this will be reported', default='1E-3')
    parser.add_argument('--mw_table', dest='mw_table',  type=pathlib.Path, required=False,
                        help='Optional yaml molecular weight table to be used instead of cantera')
    args = parser.parse_args()

    # expand any wild cards in the path name
    hdf5_paths = expand_path(args.hdf5_file)

    # Load in the cti mech file
    gas = cantera.Solution(args.mechanism_file)

    # track the max error
    maxError = 0.0

    for hdf5_file in hdf5_paths:
        # Load in the hdf5 file
        hdf5_source = h5py.File(hdf5_file, 'r')

        # get the base field to get the size
        source_field = hdf5_source["cell_fields/monitor_densityYiSource"]
        yi_field = hdf5_source["cell_fields/monitor_yi"]

        # build the list of species
        numSpecies = source_field.shape[2]
        if numSpecies != yi_field.shape[2]:
            raise Exception("The yi_field and source_field should have the same number of species")

        speciesNames = []
        for s in range(numSpecies):
            speciesNames.append(yi_field.attrs[f'componentName{s}'].decode("utf-8"))

        elementNames = gas.element_names
        numberCells = yi_field.shape[1]

        # precompute the mw
        mwi = [0] * len(speciesNames)

        if args.mw_table is None:
            # pull MW from cantera
            for s in range(numSpecies):
                mwi[s] = gas.molecular_weights[gas.species_index(speciesNames[s])]
        else:
            # load and get mw from table
            with open(args.mw_table, 'r') as mw_file:
                mw_table = yaml.safe_load(mw_file)
                # pull from mw
                for s in range(numSpecies):
                    mwi[s] = mw_table[speciesNames[s]]

        # attempt one, march over each cell and
        for c in range(numberCells):
            # size up the element change rate
            dEdt = [0] * len(elementNames)
            eTotal = [0] * len(elementNames)
            dYdt = source_field[0, c, :]

            # compute yi here
            yi = yi_field[0, c, :]
            yiTotal = 0.0

            # march over each element
            for e in range(len(elementNames)):
                elementName = elementNames[e]

                # march over each species
                for s in range(len(speciesNames)):
                    speciesName = speciesNames[s]

                    # get the ratio of atoms/species
                    ratio = gas.n_atoms(speciesName, elementName)

                    # add to the dEt/dt // assume na*vt = 1
                    dEdt[e] += ratio * dYdt[s] / mwi[s]
                    eTotal[e] += ratio * yi[s] / mwi[s]

            # march over each species
            for s in range(len(speciesNames)):
                yiTotal += yi[s]

            maxCellError = 0.0
            for e in range(len(elementNames)):
                maxCellError = max(maxCellError, abs(dEdt[e]))
            if maxCellError > maxError:
                maxError = maxCellError

            if maxCellError > args.error_report_tolerance:
                print("########################################################################################")
                print("Cell: ", c, " error: ", maxCellError)
                print("elementNames: ", elementNames)
                print("dEdt: ", dEdt)
                print("yiTotal: ", yiTotal)
                print("eTotal: ", eTotal)

    hdf5_source.close()
