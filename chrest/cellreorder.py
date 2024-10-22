# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:23:16 2024

@author: rkolo
"""

# this file is to reorganize the cells between two hdf5 output files

import argparse
import pathlib
import sys, os
import pandas

import numpy as np
import h5py
import time

from chrestData import ChrestData
from supportPaths import expand_path
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import interpolate 
from enum import Enum
from xdmfGenerator import XdmfGenerator
from multiprocessing import Pool

#support functions
def find3Dcellsectionnew(section,cellSS,vertindSS):
    indexvect=np.where((cellSS[:,0] == vertindSS[section[0]][0]) 
                                   & (cellSS[:,1] == vertindSS[section[1]][0]) 
                                   & (cellSS[:,2] == vertindSS[section[2]][0]) 
                                   & (cellSS[:,3] == vertindSS[section[3]][0])
                                   & (cellSS[:,4] == vertindSS[section[4]][0])
                                   & (cellSS[:,5] == vertindSS[section[5]][0])
                                   & (cellSS[:,6] == vertindSS[section[6]][0])
                                   & (cellSS[:,7] == vertindSS[section[7]][0]))[0][0]
    return indexvect



def find2Dcellsectionnew(args):
    section,cellSS,vertindSS=args
    # indexvect=np.zeros([section.shape[0],1])
    # for i in range(len(indexvect)):
    indexvect=np.where((cellSS[:,0] == vertindSS[section[0]][0]) 
                                   & (cellSS[:,1] == vertindSS[section[1]][0]) 
                                   & (cellSS[:,2] == vertindSS[section[2]][0]) 
                                   & (cellSS[:,3] == vertindSS[section[3]][0]))[0][0]
    return indexvect

def find2Dcellsection(args):
    section,vertindSS,SSfilePath = args
    hdf5SS = h5py.File(SSfilePath, 'r')
    cellSS=hdf5SS['/viz/topology/cells'][()]
    hdf5SS.close()
    indexvect=np.zeros([section.shape[0],1])
    for i in range(len(indexvect)):
        indexvect[i]=np.where((cellSS[:,0] == vertindSS[section[i,0]][0]) 
                                   & (cellSS[:,1] == vertindSS[section[i,1]][0]) 
                                   & (cellSS[:,2] == vertindSS[section[i,2]][0]) 
                                   & (cellSS[:,3] == vertindSS[section[i,3]][0]))[0][0]
    return indexvect


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
        
        self.vertindSS=[]
        self.cellindSS=[]

        self.savedata=False
        
    def parsedata(self):
        
        hdf5SS = h5py.File(self.SSfilePath, 'r')
        
        self.cellSS=hdf5SS['/viz/topology/cells'][()]
        self.vertSS=hdf5SS['/geometry/vertices'][()]
        self.solSS=hdf5SS['/fields/solution'][()]
        self.dimension=self.vertSS.shape[1]
        hdf5SS.close()
        
        hdf5new = h5py.File(self.newfilePath, 'r')
        self.cellnew=hdf5new['/viz/topology/cells'][()]
        self.vertnew=hdf5new['/geometry/vertices'][()]
        hdf5new.close()
        
        
        self.vertindSS=np.zeros([len(self.vertSS),1])
        self.cellindSS=np.zeros([len(self.cellSS),1])
        self.newmat=[]
        
        self.serial=True
        
        return 0
        
    def findverticies(self):
        
        if self.dimension==2:
            if self.serial:
                # serial
                start_time = time.time()
                for i in range(len(self.vertnew)):
                    self.vertindSS[i] = np.where((self.vertSS[:,0] == self.vertnew[i,0]) & (self.vertSS[:,1]==self.vertnew[i,1]))[0][0]
                print("--- %s seconds for finding the verticies 2D---" % (time.time() - start_time))
            else:
                #multiprocessing
                start_time = time.time()
                from multiprocessing import cpu_count
                n_cores = cpu_count()
                pieces = np.array_split(self.vertnew, n_cores-1)
                with Pool(processes=n_cores-1) as pool:
                    results = pool.map(self.find2Dindvertsection, [(piece) for piece in pieces])
                self.vertindSS = np.concatenate(results)  
                print("--- %s seconds for finding the verticies 2D---" % (time.time() - start_time))
        

        elif self.dimension==3:
            if self.serial:
                #serial
                start_time = time.time()
                for i in range(len(self.vertnew)):
                    self.vertindSS[i] = np.where((self.vertSS[:,0] == self.vertnew[i,0]) & (self.vertSS[:,1]==self.vertnew[i,1]) & (self.vertSS[:,2]==self.vertnew[i,2]))[0][0]
                print("--- %s seconds for finding the verticies 3D---" % (time.time() - start_time))
            else:
                #multiprocessing
                start_time = time.time()
                from multiprocessing import cpu_count
                n_cores = cpu_count()
                
                pieces = np.array_split(self.vertnew, n_cores-1)
                # for piece in pieces:
                #     a=self.findindvertsection(piece)
                # Create a pool of workers
                with Pool(processes=n_cores-1) as pool:
                    results = pool.map(self.find3Dindvertsection, [(piece) for piece in pieces])
                self.vertindSS = np.concatenate(results)  
                print("--- %s seconds for finding the verticies 3D---" % (time.time() - start_time)) 
            
        return 0     
    
    def findcells(self):
        from multiprocessing import cpu_count
        n_cores = cpu_count()
        if self.dimension==2:
            
            if self.serial:
                
                #option 1 old way
                start_time = time.time()
                for i in range(len(self.cellSS)):
                    self.cellindSS[i]=np.where((self.cellSS[:,0] == self.vertindSS[self.cellnew[i,0]][0]) 
                                               & (self.cellSS[:,1] == self.vertindSS[self.cellnew[i,1]][0]) 
                                               & (self.cellSS[:,2] == self.vertindSS[self.cellnew[i,2]][0]) 
                                               & (self.cellSS[:,3] == self.vertindSS[self.cellnew[i,3]][0]))[0][0]
                print("--- %s seconds for finding the cells serial ---" % (time.time() - start_time))
            
            
            else:
                #option 2 slit it up 
                start_time = time.time()
                pieces = np.array_split(self.cellnew, n_cores-1)
                # Create a pool of workers
                # for row in self.cellnew:
                #     a=self.find2Dcellsectionnew(row)
                with Pool(processes=n_cores-1) as pool:
                    results = pool.map(self.find2Dcellsectionclass, [piece for piece in pieces])
                    # results = pool.map(self.find2Dcellsectionnew, [row for row in self.cellnew])
                self.cellindSS = np.concatenate(results) 
                # self.cellindSS = results
                print("--- %s seconds for finding the cells class pieces ---" % (time.time() - start_time))            
            
            
            # #option 3 give the entire array to map
            # start_time = time.time()
            # pieces = np.array_split(self.cellnew, n_cores-1)
            # # Create a pool of workers
            # # for row in self.cellnew:
            # #     a=self.find2Dcellsectionnew(row)
            # with Pool(processes=n_cores-1) as pool:
            #     # results = pool.map(self.find2Dcellsection, [piece for piece in pieces])
            #     results = pool.map(self.find2Dcellsectionnewclass, [row for row in self.cellnew])
            # # self.cellindSS = np.concatenate(results) 
            # self.cellindSS = results
            # print("--- %s seconds for finding the cells class rows ---" % (time.time() - start_time))



            # #option 4 slit it up 
            # start_time = time.time()
            # pieces = np.array_split(self.cellnew, n_cores-1)
            # # Create a pool of workers
            # # for row in self.cellnew:
            # # a=find2Dcellsection([pieces[0],self.cellSS,self.vertindSS])
            # # a=self.cellSS
            # # b=self.vertindSS
            # with Pool(processes=n_cores-1) as pool:
            #     results = pool.map(find2Dcellsection, [(piece,self.vertindSS,self.SSfilePath) for piece in pieces])
            #     # results = pool.map(find2Dcellsection, [(piece,a,b) for piece in pieces])

            # self.cellindSS = np.concatenate(results) 
            # # self.cellindSS = results
            # print("--- %s seconds for finding the cells pieces ---" % (time.time() - start_time))


            # #option 5 give the entire array to map
            # start_time = time.time()
            # pieces = np.array_split(self.cellnew, n_cores-1)
            # # Create a pool of workers
            # # for row in self.cellnew:
            # #     a=self.find2Dcellsectionnew(row)
            # with Pool(processes=n_cores-1) as pool:
            #     # results = pool.map(self.find2Dcellsection, [piece for piece in pieces])
            #     results = pool.map(find2Dcellsectionnew, [(row,self.cellSS,self.vertindSS) for row in self.cellnew])
            # # self.cellindSS = np.concatenate(results) 
            # self.cellindSS = results
            # print("--- %s seconds for finding the cells rows---" % (time.time() - start_time))

        elif self.dimension==3:
            
            if self.serial:
            
                #option 1 - serial
                start_time = time.time()
                # for i in range(len(self.cellSS)):
                for i in range(len(self.cellSS)):
                    self.cellindSS[i]=np.where((self.cellSS[:,0] == self.vertindSS[self.cellnew[i,0]][0]) 
                                               & (self.cellSS[:,1] == self.vertindSS[self.cellnew[i,1]][0]) 
                                               & (self.cellSS[:,2] == self.vertindSS[self.cellnew[i,2]][0]) 
                                               & (self.cellSS[:,3] == self.vertindSS[self.cellnew[i,3]][0])
                                               & (self.cellSS[:,4] == self.vertindSS[self.cellnew[i,4]][0])
                                               & (self.cellSS[:,5] == self.vertindSS[self.cellnew[i,5]][0])
                                               & (self.cellSS[:,6] == self.vertindSS[self.cellnew[i,6]][0])
                                               & (self.cellSS[:,7] == self.vertindSS[self.cellnew[i,7]][0]))[0][0]
                print("--- %s seconds for finding the cells serial ---" % (time.time() - start_time))
            
            
            
            #option 2 -multiprocessing
            
            else:
                start_time = time.time()
                pieces = np.array_split(self.cellnew, n_cores-1)
                # Create a pool of workers
                # for row in self.cellnew:
                #     a=self.find2Dcellsectionnew(row)
                with Pool(processes=n_cores-1) as pool:
                    # results = pool.map(self.find2Dcellsection, [piece for piece in pieces])
                    results = pool.map(find3Dcellsectionnew, [(row,self.cellSS,self.vertindSS) for row in self.cellnew])
                # self.cellindSS = np.concatenate(results) 
                self.cellindSS = results
                print("--- %s seconds for finding the cells multi ---" % (time.time() - start_time))
            
            
        return 0

        
    def reorder(self):
        self.newmat=np.zeros_like(self.solSS)
        for i in range(len(self.cellSS)):
            self.newmat[0,i,:]=self.solSS[0,int(self.cellindSS[i]),:]
        return 0
    
    def writedata(self):
        
        hdf5new = h5py.File(self.newfilePath, 'r+')

        solnew=hdf5new['/fields/solution'][()]
        solnew[...] = self.newmat
        
        hdf5new['/fields/solution'][()]=self.newmat
        
        hdf5new.close()

        
        hdf5new = h5py.File(self.newfilePath, 'r+')
        b=hdf5new['/fields/solution'][()]
        # solnew[...] = self.newmat
        hdf5new.close()
        
        return 0
    
    def find2Dindvertsection(self,section):
        indexvect=np.zeros([section.shape[0],1])
        for i in range(len(indexvect)):
            indexvect[i] = np.where((self.vertSS[:,0] == section[i,0]) & (self.vertSS[:,1]==section[i,1]))[0][0]
        return indexvect  
    
    def find3Dindvertsection(self,section):
        indexvect=np.zeros([section.shape[0],1])
        for i in range(len(indexvect)):
            indexvect[i] = np.where((self.vertSS[:,0] == section[i,0]) & (self.vertSS[:,1]==section[i,1])& (self.vertSS[:,2]==section[i,2]))[0][0]
        return indexvect        

    def find2Dcellsectionnewclass(self,section):
        # indexvect=np.zeros([section.shape[0],1])
        # for i in range(len(indexvect)):
        indexvect=np.where((self.cellSS[:,0] == self.vertindSS[section[0]][0]) 
                                       & (self.cellSS[:,1] == self.vertindSS[section[1]][0]) 
                                       & (self.cellSS[:,2] == self.vertindSS[section[2]][0]) 
                                       & (self.cellSS[:,3] == self.vertindSS[section[3]][0]))[0][0]
        return indexvect
    
    def find2Dcellsectionclass(self,section):
        indexvect=np.zeros([section.shape[0],1])
        for i in range(len(indexvect)):
            indexvect[i]=np.where((self.cellSS[:,0] == self.vertindSS[section[i,0]][0]) 
                                       & (self.cellSS[:,1] == self.vertindSS[section[i,1]][0]) 
                                       & (self.cellSS[:,2] == self.vertindSS[section[i,2]][0]) 
                                       & (self.cellSS[:,3] == self.vertindSS[section[i,3]][0]))[0][0]
        return indexvect
    
    def find3Dcellsection(self,section):
        indexvect=np.zeros(section.shape)
        for i in range(len(indexvect)):
            indexvect[i]=np.where((self.cellSS[:,0] == self.vertindSS[section[i,0]][0]) 
                                       & (self.cellSS[:,1] == self.vertindSS[section[i,1]][0]) 
                                       & (self.cellSS[:,2] == self.vertindSS[section[i,2]][0]) 
                                       & (self.cellSS[:,3] == self.vertindSS[section[i,3]][0])
                                       & (self.cellSS[:,4] == self.vertindSS[section[i,4]][0])
                                       & (self.cellSS[:,5] == self.vertindSS[section[i,5]][0])
                                       & (self.cellSS[:,6] == self.vertindSS[section[i,6]][0])
                                       & (self.cellSS[:,7] == self.vertindSS[section[i,7]][0]))[0][0]
        return indexvect
        

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
    
    convert.parsedata()
    convert.findverticies()
    convert.findcells()
    convert.reorder()
    convert.writedata()
    
    print("Finished reordering the fields")
    
