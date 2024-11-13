# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:23:33 2024

@author: rkolo
"""

# File convert
#The aim of this file is to convert between two files using nearest nodes


import argparse
import pathlib
from sys import exit
import numpy as np
import h5py
from scipy.spatial import KDTree


class Fieldconvert:
    """
    Creates a new class from hdf5 chrest formatted file(s).
    """

    def __init__(self):

        self.dimension=0
        
        self.SSfilePath=[]
        self.cellSS=[]
        self.vertSS=[]
        self.solSS=[]
        
        self.newfilePath=[]
        self.cellnew=[]
        self.vertnew=[]
        self.solnew=[]
        self.points=[]
        
        self.vertindSS=[]
        self.cellindSS=[]

        self.savedata=False
        
    def parsedata(self):
        
        hdf5SS = h5py.File(self.SSfilePath, 'r')
        
        self.cellSS=hdf5SS['/viz/topology/cells'][()]
        self.vertSS=hdf5SS['/geometry/vertices'][()]
        self.solSS=hdf5SS['/fields/solution'][()]
        self.speciesSS = [None] * hdf5SS['/cell_fields/solution_densityYi'].shape[2]
        # Determine the species
        for k in hdf5SS['/cell_fields/solution_densityYi'].attrs.keys():
            print(f"{k} => {hdf5SS['/cell_fields/solution_densityYi'].attrs[k]}")
            if isinstance(hdf5SS['/cell_fields/solution_densityYi'].attrs[k],np.bytes_):
                #get the order right...
                if k[0:13]=='componentName':
                    idx=''.join(filter(lambda i: i.isdigit(), k))
                    self.speciesSS[int(idx)]=hdf5SS['/cell_fields/solution_densityYi'].attrs[k].decode('UTF-8')
        
        
        self.dimensionSS=self.vertSS.shape[1]
        hdf5SS.close()
        
        hdf5new = h5py.File(self.newfilePath, 'r')
        self.cellnew=hdf5new['/viz/topology/cells'][()]
        self.vertnew=hdf5new['/geometry/vertices'][()]
        self.solnew=hdf5new['/fields/solution'][()]
        self.speciesnew = [None] * hdf5new['/cell_fields/solution_densityYi'].shape[2]
        # Determine the species
        for k in hdf5new['/cell_fields/solution_densityYi'].attrs.keys():
            print(f"{k} => {hdf5new['/cell_fields/solution_densityYi'].attrs[k]}")
            if isinstance(hdf5new['/cell_fields/solution_densityYi'].attrs[k],np.bytes_):
                #get the order right...
                if k[0:13]=='componentName':
                    idx=''.join(filter(lambda i: i.isdigit(), k))
                    self.speciesnew[int(idx)]=hdf5new['/cell_fields/solution_densityYi'].attrs[k].decode('UTF-8')
        self.dimensionnew=self.vertnew.shape[1]
        hdf5new.close()
        
        self.newmat=[]
        
        #checking if the mechanisms are the same
        self.samemech=True
        for i,x in enumerate(self.speciesSS):
            self.samemech=x==self.speciesnew[i]
            
            
        #TODO change this in the future at some point...
        if not self.samemech:
            raise Exception("I can only do the same mechnisms for now...")
            exit()
            
        if self.dimensionSS != self.dimensionnew:
            print('The dimensions of the two files dont match. The only thing currently supported is a 3D->3D and 2D->3D 2D->2D')
            
            if self.dimensionSS ==3:
                raise Exception('The steady state file has 3 dimensions')
                exit()
            else:
                self.addzmom=True
            
        if (self.dimensionnew+2+len(self.speciesnew)!=self.solnew.shape[2]) == (self.dimensionnew+2+len(self.speciesnew)!=self.solnew.shape[2]) :
            raise Exception('Only one of the files has extra variables bubba...')
            exit()
            

        return 0
    
    def compute_cell_centers(self,cells,vertices, dimensions=-1):
        # create a new np array based upon the dim
        number_cells = cells.shape[0]
        # vertices = vertices[:]
        vertices_dim = 1
        if len(vertices.shape) > 1:
            vertices_dim = vertices.shape[1]
        if dimensions < 0:
            dimensions = vertices_dim

        coords = np.zeros((number_cells, dimensions))

        # march over each cell
        for c in range(len(coords)):
            cell_vertices = vertices.take(cells[c], axis=0)

            # take the average
            cell_center = np.sum(cell_vertices, axis=0)
            cell_center = cell_center / len(cell_vertices)

            # put back
            coords[c, 0:vertices_dim] = cell_center

        # if this is one d, flatten
        if dimensions == 1:
            coords = coords[:, 0]

        return coords
        
    def findcells(self):

        cell_centersSS = self.compute_cell_centers(self.cellSS,self.vertSS[:],self.dimensionSS)
        cell_centersnew = self.compute_cell_centers(self.cellnew,self.vertnew[:],self.dimensionnew)
        
        if cell_centersSS.shape[1] != cell_centersnew.shape[1]:
        
            if cell_centersSS.shape[1] == 2 and cell_centersnew.shape[1] == 3:
                print('SS shape is 2D and new shape is 3D. The code is assuming the 2D mesh is on the xy-plane')
                b=np.zeros(cell_centersSS.shape[0])
                cell_centersSS=np.append(cell_centersSS,b.reshape(b.shape[0],1),1)           
        
        tree = KDTree(cell_centersSS)
        dist, self.points = tree.query(cell_centersnew, workers=-1)
            
        return 0

        
    def reorder(self):
        self.newmat=np.zeros_like(self.solnew)
        
        if self.addzmom:
            for i in range(len(self.cellnew)):
                # print(self.points[i])
                #assume z momentum is 0
                self.newmat[0,i,:]=np.insert(self.solSS[0,self.points[i],:],4,0)
            return 0
        else:
            #for simple 3D->3D same mechanism
            for i in range(len(self.cellnew)):
                # print(self.points[i])
                self.newmat[0,i,:]=self.solSS[0,self.points[i],:]
            return 0

    
    def writedata(self):
        
        hdf5new = h5py.File(self.newfilePath, 'r+')

        solnew=hdf5new['/fields/solution'][()]
        solnew[...] = self.newmat
        
        hdf5new['/fields/solution'][()]=self.newmat
        
        hdf5new.close()
      
        return 0
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a chrest data file from an ablate file')

    parser.add_argument('--fileSS', dest='fileSS', type=pathlib.Path, required=True,
                            help='The steady state file path'
                                 'to supply more than one file.')
        
    parser.add_argument('--filenew', dest='filenew', type=pathlib.Path, required=True,
                            help='The steady state file path'
                                 'to supply more than one file.')
    
    parser.add_argument('--parallel', dest='parallel', action='store_true',
                        help="running it in parallel",
                        )

    print("Start reordering the fields")
    args = parser.parse_args()
    
    convert=Fieldconvert()
    
    if args.parallel:
        convert.serial=False
    
    convert.SSfilePath=args.fileSS
    convert.newfilePath=args.filenew
    
    #Step 1 parse data
    convert.parsedata()
    
    #Step 2 Calculate the cell centers, make tree, find closest
    convert.findcells()
        
    #Step 3 Reorder the data
    convert.reorder()
    
    #Step 4 Write out files
    convert.writedata()
    
    print("Finished reordering the fields")
