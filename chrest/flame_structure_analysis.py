#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 07:48:00 2023

@author: ped3
"""

if __name__ == "__main__":
   
   from chrestData import *
   
   minimum_time=0.016
   maximum_time = 0.033
#   maximum_time = 1.
   indzplane = 25
   
    # load Temperature 
   path = '/Users/ped3/Scratch/PMMA_Feb2_.153m3_min/Ablate/6G-186x40-pmma-rad-12906393/domain.chrest/' 
   domain = ChrestData(path+'domain.*.hdf5')
   T,time,components = domain.get_field('temperature',min_time=minimum_time,max_time=maximum_time)
   print('temperature loaded.')
   
   # load mixture fraction related files
   path = '/Users/ped3/Scratch/PMMA_Feb2_.153m3_min/Ablate/6G-186x40-pmma-rad-12906393/flowField_mixtureFraction.chrest/'
   flow = ChrestData(path+'flowField_mixtureFraction.chrest.*.hdf5')
   zmix, time, components = flow.get_field('zMix',min_time=minimum_time,max_time=maximum_time)
   print('Zmix loaded.')
   Yi, time, components = flow.get_field('Yi',min_time=minimum_time,max_time=maximum_time)
   print('Yi loaded.')
   Yindx = dict()
   for c in components:
     Yindx[c]=components.index(c) 
   
   # grid information
   grid = flow.get_coordinates()
   x = grid[0,0,:,0]
   y = grid[0,:,0,1]
   z = grid[:,0,0,2]
   nx = len(x)
   ny = len(y)
   nz = len(z)
   
   # Scatter plot comparisons using flamelets

   from XML1DFlameReader import ReadFile
   
   OneDFile ="oneDflames_Twall=653.0NoSoot_.xml"
   [flames,flamekeys] = ReadFile(OneDFile)
        #Sort the flamekeys into an array of decreasing length
   def doubleval(e): return float(e)
   flamekeys.sort(key=doubleval, reverse=True)

  #  three indices from 1D flames  representing:
  #       indRQ: last flame before Radiative Quenching
  #       indMT: flame with Max Temperature
  #       indBO: last flame before high strain extinction
   indRQ,indMT,indBO = [np.nan,np.nan,np.nan]; ExtinguishTemp = 800;
   idx = 0; # variable to track idx of flamekey
   OverallMaxT = 0; #Variable to track maximum temperature in all flames
   SSFlamesFound = False; #Variable to track if we have encountered a S.S. lit flame yet
   for f in flamekeys:
         maxT = np.max(flames[f]['T'])

         if(maxT > ExtinguishTemp):
             #Currently have a S.S flame
             #Check if this is the first one
             if(np.isnan(indRQ)):
                 indRQ = idx
             #Check if this is the flame with the current maximum temp overall
             if(maxT > OverallMaxT):
                 OverallMaxT = maxT
                 indMT = idx
             SSFlamesFound = True;
         else:
             #Check if this is the first flamekey after the flame have been found
             if(SSFlamesFound):
                 indBO = idx - 1
                 SSFlamesFound = False
         idx = idx +1


    # plot flame structure comparisons
   import matplotlib.pyplot as plt
   linesty = ['-',':','--','-.','-']
   col = ['red','blue','green','purple']
   mar = ['o','s','^','D','+']  
   indArray = [indRQ,indMT,indBO];
   indNames = ["Radiative Quenching","Maximum Temperature", "Blow Off"]
   
      # T
   f, ax1 = plt.subplots(1,1)
   ax1.set_xlabel(r'$ Z $',size=16)
   ax1.set_ylabel('$T(K)$',size=16)
    # scatter from DNS slice
   zscat = zmix[:,indzplane,:,:].flatten("C")
   Tscat = T[:,indzplane,:,:].flatten("C")
   ax1.plot(zscat,Tscat,color = 'black',linestyle='None',marker=',',markersize=1)   
   # strained flames
   ind=0
   for indf in indArray:
       # strained flames
        Zmix1D = flames[flamekeys[indf]]['Zmix']
        T1D = flames[flamekeys[indf]]['T']
        ax1.plot(Zmix1D,T1D,color = col[ind],label=indNames[ind],lw =3)
        ind=ind+1       
   ax1.grid(True)
   plt.ylim(300,3200)
   ax1.legend(loc=0)
   plt.savefig('Tscat.png',bbox_inches='tight')
   plt.show()
   
   # CO2 
   f, ax1 = plt.subplots(1,1)
   ax1.set_xlabel(r'$ Z $',size=16)
   ax1.set_ylabel('$Y_{CO_2}$',size=16)
    # scatter from DNS slice
   zscat = zmix[:,indzplane,:,:].flatten("C")
   Yiscat = Yi[:,indzplane,:,:,Yindx['CO2']].flatten("C")
   ax1.plot(zscat,Yiscat,color = 'black',linestyle='None',marker=',',markersize=1)   
   # strained flames
   ind=0
   for indf in indArray:
       # strained flames
        Zmix1D = flames[flamekeys[indf]]['Zmix']
        Yi1D = flames[flamekeys[indf]]['YiCO2']
        ax1.plot(Zmix1D,Yi1D,color = col[ind],label=indNames[ind],lw =3)
        ind=ind+1       
   ax1.grid(True)
   ax1.legend(loc=0)
   plt.savefig('CO2scat.png',bbox_inches='tight')
   plt.show()
   
      # H2O
   f, ax1 = plt.subplots(1,1)
   ax1.set_xlabel(r'$ Z $',size=16)
   ax1.set_ylabel('$Y_{H_2O}$',size=16)
    # scatter from DNS slice
   zscat = zmix[:,indzplane,:,:].flatten("C")
   Yiscat = Yi[:,indzplane,:,:,Yindx['H2O']].flatten("C")
   ax1.plot(zscat,Yiscat,color = 'black',linestyle='None',marker=',',markersize=1)   
   # strained flames
   ind=0
   for indf in indArray:
       # strained flames
        Zmix1D = flames[flamekeys[indf]]['Zmix']
        Yi1D = flames[flamekeys[indf]]['YiH2O']
        ax1.plot(Zmix1D,Yi1D,color = col[ind],label=indNames[ind],lw =3)
        ind=ind+1       
   ax1.grid(True)
   ax1.legend(loc=0)
   plt.savefig('H2Oscat.png',bbox_inches='tight')
   plt.show()
   
    # MMA (fuel)
   f, ax1 = plt.subplots(1,1)
   ax1.set_xlabel(r'$ Z $',size=16)
   ax1.set_ylabel('$Y_{MMA}$',size=16)
    # scatter from DNS slice
   zscat = zmix[:,indzplane,:,:].flatten("C")
   Yiscat = Yi[:,indzplane,:,:,Yindx['MMETHAC_C5H8O2']].flatten("C")
   ax1.plot(zscat,Yiscat,color = 'black',linestyle='None',marker=',',markersize=1)   
   # strained flames
   ind=0
   for indf in indArray:
       # strained flames
        Zmix1D = flames[flamekeys[indf]]['Zmix']
        Yi1D = flames[flamekeys[indf]]['YiMMETHAC_C5H8O2']
        ax1.plot(Zmix1D,Yi1D,color = col[ind],label=indNames[ind],lw =3)
        ind=ind+1       
   ax1.grid(True)
   ax1.legend(loc=0)
   plt.savefig('MMAscat.png',bbox_inches='tight')
   plt.show()
   
         # OH
   f, ax1 = plt.subplots(1,1)
   ax1.set_xlabel(r'$ Z $',size=16)
   ax1.set_ylabel('$Y_{OH}$',size=16)
    # scatter from DNS slice
   zscat = zmix[:,indzplane,:,:].flatten("C")
   Yiscat = Yi[:,indzplane,:,:,Yindx['OH']].flatten("C")
   ax1.plot(zscat,Yiscat,color = 'black',linestyle='None',marker=',',markersize=1)   
   # strained flames
   ind=0
   for indf in indArray:
       # strained flames
        Zmix1D = flames[flamekeys[indf]]['Zmix']
        Yi1D = flames[flamekeys[indf]]['YiOH']
        ax1.plot(Zmix1D,Yi1D,color = col[ind],label=indNames[ind],lw =3)
        ind=ind+1       
   ax1.grid(True)
   ax1.legend(loc=0)
   plt.savefig('OHscat.png',bbox_inches='tight')
   plt.show()
   
            # CO
   f, ax1 = plt.subplots(1,1)
   ax1.set_xlabel(r'$ Z $',size=16)
   ax1.set_ylabel('$Y_{CO}$',size=16)
    # scatter from DNS slice
   zscat = zmix[:,indzplane,:,:].flatten("C")
   Yiscat = Yi[:,indzplane,:,:,Yindx['CO']].flatten("C")
   ax1.plot(zscat,Yiscat,color = 'black',linestyle='None',marker=',',markersize=1)   
   # strained flames
   ind=0
   for indf in indArray:
       # strained flames
        Zmix1D = flames[flamekeys[indf]]['Zmix']
        Yi1D = flames[flamekeys[indf]]['YiCO']
        ax1.plot(Zmix1D,Yi1D,color = col[ind],label=indNames[ind],lw =3)
        ind=ind+1       
   ax1.grid(True)
   ax1.legend(loc=0)
   plt.savefig('COscat.png',bbox_inches='tight')
   plt.show()
        
              
       
        
  
   
   
   
   