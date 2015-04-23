#verarching Script

import numpy
import tables
import scipy
import matplotlib
import matplotlib.pyplot as plt
import collections
import scipy.signal
import csv
import egan_vorpalUtil as egan
import os
#import plotPlasmoids as pplas
import pplot
DEBUG=False
RunOne=False #if I just want to run one dump
RunNone=0
DistPlots=0
dumpOne=27
Plots=True
#This

#widths=[1,5,15,30,40]
widths=[15]
##Simulation Data Info
#simNums=[1,2,3,4]
simNums=[1]
#widths=[40]
for simNum in simNums:
    #simNum=1
    print "simNum %d"%simNum
    for width in widths:
        print "width: %d"%width
	runName="relRecon2p_AzReduced_"
	if simNum==1:
		pathName="/verus/wernerg/edisonData/initPowLaw/"
	else:
		pathName="/scr_verus/wernerg/vrun/relRecon/plasmoids/m1-400-sig30-thd1p875-thbp01-res2-ppc32-"+str(simNum)+"/"
	peakWidth=width#defines how wide the max/min has to be to be considered a true max/min
	numCells=1600
	simLabel="init_powerlaw"
	totalDumps=80
	ypos=numCells/4
	yposT=numCells*3/4
	print "y positions:"
	print ypos
	print yposT
	width=peakWidth
	minWidth=width
	data_headers=["stepnum","plasmoidnum","ypos","xloc","phimax","xL","xR","phiL","phiR","phimin","xmin","ytop", "ybot","time","stage"]
	##Out file names
	resultsDir="./"+str(simLabel)+"_width_"+str(width)+"/"
	if not os.path.exists(resultsDir):
	    os.makedirs(resultsDir)
#	if RunNone==False:
#		extendedName=egan.startExtended(resultsDir, runName+"width_"+str(width), data_headers)
#		extendedNameT=egan.startExtended(resultsDir,runName+"width_"+str(width)+'T',data_headers)      
#		shortName=egan.startShort(resultsDir,runName+"width_"+str(width))  
		#print extendedName
        extendedName="init_powerlaw_width_15/relRecon2p_AzReduced_width_15_extended.csv"
	extendedNameT="init_powerlaw_width_15/relRecon2p_AzReduced_width_15T_extended.csv"
	execfile("/verus/wernerg/edisonData/initPowLaw/relReconVars.py")
	# I'll figure out a better way to do deal with data_headers smoothly at some point
	AVG=False
	if RunNone==False:
	    for i in range(80,120):
		dump=i
		if RunOne:
		    print "running just one!"
		    dump=dumpOne
		print "StepNum= %d"%(dump)   
		filename=pathName+"reducedAz/"+runName+str(dump)+".h5"
		print filename
		Az=egan.makeAz(filename,REDUCED=True)
		#print Az
		#print Az.shape()
		numCells=Az.shape
		#print "Numcells test"
		#print numCells
       		ypos=numCells[0]/4
       		yposT=numCells[0]*3/4
		#print ypos
		maxima=egan.findAzMax(Az,ypos,peakWidth,AVG=AVG)
		#print maxima
		minima=egan.findAzMin(Az,ypos,minWidth,AVG=AVG)
		maxT=egan.findAzMax(abs(Az),yposT,peakWidth,AVG=AVG)
		minT=egan.findAzMin(abs(Az),yposT,minWidth,AVG=AVG)
		plotname=resultsDir+runName+"MaxMinPlot_width"+str(width)+"_step"+str(dump)+".eps"
		plotnameT=resultsDir+runName+"MaxMinPlot_width"+str(width)+"_top_step_"+str(dump)+".eps"
		contourplotname=resultsDir+runName+"plasmoidOutline_width"+str(width)+"_testTB"+str(dump)+".eps"
		if len(maxima)<2: #Break if we get to the steady state non-physical soln
		    print "End of steps"
		    break
		elif len(maxT)<2:
		    print "End of steps (from top)"
		    break

		time,stage=egan.getTime(filename, pathName)
		#print time
		#print stage
		pArray=egan.plasmoids_2nd(ypos,Az,maxima,minima,runName,dump,time,stage,heightParam=0)
		pArrayT=egan.plasmoids_2nd(yposT,abs(Az),maxT,minT,runName+'_T',dump,time,stage,heightParam=0)
		if Plots:
		   egan.plotAzMaxMin(Az,pArray,ypos,maxima,minima,dump,plotname)
		   egan.plotAzMaxMin(abs(Az),pArrayT,yposT,maxT,minT,dump,plotnameT)
		   egan.plotPlasmoidContoursWTop(ypos,yposT,Az,maxima,minima,maxT,minT,pArray,pArrayT,dump,contourplotname,DEBUG=DEBUG,ZOOM_SWITCH=False) #saves time to cut out

		print "time= %f"%time
		egan.writeExtended(pArray,extendedName,dump)
		egan.writeExtended(pArrayT,extendedNameT,dump)
		#egan.writeShort(pArray,shortName,dump)
		#egan.writeShort(pArrayT,shortName,dump)
		if RunOne: #end after just one dump if RunOne set
		    print "Breaking now due to RunOne"
		    break
	if DistPlots:
	    resultsDirs=[]
            #width=15
            flux=1
            print "Results dir %s"%resultsDir
	    for simNum in simNums:
		simLabel=str(numCells[1])+"_"+str(simNum)
		resultsDirs.append("./"+str(simLabel)+"_width_"+str(width)+"/")
	    print resultsDirs
	    #pplot.dist_1d(width, pathName, resultsDir, simLabel,NX_TOTS, DELTA, LX,B_0,resultsDirs,FLUX=flux,SAVE_TOGETHER=1,MULT_RUNS=True,STAGES=3)
#	    pplot.dist_1d(width, pathName, resultsDir, simLabel,NX_TOTS, DELTA, LX,B_0,resultsDirs,WIDTH=True,SAVE_TOGETHER=1,MULT_RUNS=True)
#	    pplot.dist_2d(width, pathName, resultsDir, simLabel,NX_TOTS, DELTA, LX,B_0,TIME_PLOT=True,SAVE_TOGETHER=1)
	    pplot.dist_2d(width, pathName, resultsDir, simLabel,NX_TOTS, DELTA, LX,B_0,STAGES=False,SAVE_TOGETHER=0)
#	    pplot.dist_2d(width, pathName, resultsDir, simLabel,NX_TOTS, DELTA, LX,B_0,SAVE_TOGETHER=1)
