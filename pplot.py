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
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import sys
#sys.path.append('/scr_verus/wernerg/vrun/relRecon/relReconRepo')
import calcStats
import vorpalUtil
####switch to function that will read into arrays passed out, then update 1d dists to have stage info
import pplot
def read_plasmoids(width, \
		pathName,\
		resultsDir, \
		simLabel,\
		NX_TOTS,\
		DELTA,\
		LX,\
		B_0):
    runName="relRecon2p_"
    widths=[]
    deltapsi=[]
    dump=[]
    xwidth=[]
    time=[]
    xpos=[]
    stage=[]
    wT=[] #top width
    wB=[]#bottom width
    for location in ["b","t"]:
        if location=="b":
            filename=resultsDir+runName+"width_"+str(width)+"_extended.csv"
        else:
                filename=resultsDir+runName+"width_"+str(width)+"T_extended.csv"
        print location
        print filename
        with open(filename,'r') as csvfile:
                freader=csv.reader(csvfile,dialect='excel')
                head1=next(freader)
                columns=next(freader)
                for row in freader:
                        widths.append(float(row[11])/2+float(row[12])/2)
                        deltapsi.append(abs(float(row[4])-float(row[9])))
                        xwidth.append(abs(float(row[3])-float(row[10])))
                        dump.append(row[0])
                        xpos.append(row[3])
                        time.append(row[13])
                        stage.append(row[14])
			wT.append(row[11])
			wB.append(row[12])
    
    ####################################
    # Convert arrays to proper form      ###
    ####################################         
    #print dump
    #execfile(pathName+"relReconVars.py") 
    dump= numpy.array(dump,dtype=int)
    wT=numpy.array(wT,dtype=float)
    wB=numpy.array(wB,dtype=float)
    time=numpy.array(time,dtype=float)
    xpos=numpy.array(xpos,dtype=int)
    stage=numpy.array(stage,dtype=int)
    widthNorm=[abs(float(i))/NX_TOTS[0] for i in widths]
    fluxNorm=[abs(float(i))/(B_0*LX) for i in deltapsi]
    xwidthNorm=[abs(float(i)/NX_TOTS[0]) for i in xwidth]
  
    return fluxNorm,widthNorm,xpos,dump,time,stage,wT,wB 



def dist_2d(width, \
		pathName,\
		resultsDir,\
		simLabel,\
		NX_TOTS,\
		DELTA,\
		LX,\
		B_0,\
		STAGES=False,\
		TIME_PLOT=False,\
		SAVE_TOGETHER=False):
    print "2D Dist Plotting"
    plotDir=resultsDir+"distplots/"
    if SAVE_TOGETHER:
	plotDir="today_distplots/"
    runName="relRecon2p_"
    origfilename=pathName+runName+'yeeB_'+str(1)+".h5"
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
#    print filename
    fluxNorm,widthNorm,xpos,dump,time,stage,wT,wB=pplot.read_plasmoids(width,pathName,resultsDir,simLabel,NX_TOTS,DELTA, LX,B_0)
#find points that I want to exclude
    shift=[]
#    print wT[1]
#    print max(wT[1],wB[1])
    for i in range(len(wT)):
	    shift.append(abs(wT[i]-wB[i])/max(wT[i],wB[i]))
#    print shift
    
    #########################
    # Get shorter time steps
    ########################
    if TIME_PLOT:
	simName=pathName+"relRecon2p"
	dnByName = "layerDnByLine"
	(dnBy, c, dnByLbs, dnByUbs) = vorpalUtil.getFieldArrayHistory(simName, dnByName)
	byHistTimes = vorpalUtil.getHistoryTimes(simName, dnByName)[:,0]
	(ndim, numPhysCells, startCell, lowerBounds, upperBounds) = vorpalUtil.getSimGridInfo(simName)
	#print "same as NX_TOTS?" 
	#print numPhysCells
	dxs = (upperBounds-lowerBounds) / numPhysCells
	dx=dxs
	dz=dxs

	test=dnBy.cumsum(axis=1)*-dx[0]*dz[0]
	Az=numpy.array([test[i,:,0]+.1 for i in numpy.arange(len(dnBy))])
    #######################
    ## SET UP PLOT    #####
    #######################
    if STAGES:
    	plotfile="2Ddist_stages_"+simLabel+"_width_"+str(width)
    	title="2D Plasmoid Dist-by Stages,"+simLabel+", width="+str(width)
    else:
    	title="2D Plasmoid Dist,"+simLabel+", width="+str(width)
    	plotfile="2Ddist"+simLabel+"_width_"+str(width)	
    ylabel="Normalized y half-widths (wy/L)" #default val
    xlabel="Normalized enclosed flux (deltaPsi/B_0*L)" #default val
    if TIME_PLOT:
	plotfile="time_evolution_"+simLabel+"_width_"+str(width)
	title="'O' pts in time, "+simLabel+", width="+str(width)
	ylabel="Time (1E-6 s)"
	xlabel="X-position of 'O' pts"
    
    colors=["b","g","r","c","m","y"]
    markerstyle=['.','o','v','^','<','>','s','p','*','d','D']
    markerstyle=markerstyle*(max(dump)/len(markerstyle)+1)
    colors=colors*(max(dump)/len(colors)+1)
    #print colors
    ax = host_subplot(111, axes_class=AA.Axes)
    y2=ax.twinx() #For 2nd axes
    ax.grid(True)
    if TIME_PLOT:
    	ax.set_xlim(0,800)
 	ax.set_ylim(0,max(time)*1.2)
 	y2.set_ylabel("Dump number") 
    	y2.set_ylim(0,max(dump)*1.2)
    	y2.axis["right"].label.set_color('k')
    else:
	ax.set_xlim(10**-5,1)
    	ax.set_ylim(10**-5,1)
    	y2.set_ylabel("Number of Cells")
    	y2.set_yscale("log")
    	y2.set_ylim(10**(-5)*NX_TOTS[0],1*NX_TOTS[0])
    	y2.axis["right"].label.set_color('k')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    #SET RIGHT AXIS

    #plot extra lines
    if TIME_PLOT==False:
	x=numpy.arange(1,10**5,10**2)
        x=x*(10**-5)
   	ax.loglog([.000005,1],[NX_TOTS[0]**-1,NX_TOTS[0]**-1],linestyle='--',color='k') ##Plot line of cell siz
    	ax.loglog([.000005,1],[DELTA/LX,DELTA/LX],linestyle='-',color='k') ##Plot line of reconnection layer width
    	ax.loglog(B_0*DELTA*numpy.log(numpy.cosh(x*NX_TOTS[0]/DELTA)),x,linestyle='-') #plot of background mag profile

    ##########
    #  Plot!
    ##########
    shift=numpy.array(shift, dtype=float)
    #print max(stage)
    #print range(1,max(stage))
    test2=numpy.where(shift<.1)
    print len(test2[0])
    print len(shift)
    print float(len(test2[0]))/float(len(shift))

    if STAGES:
      for i in range(1,max(stage)+1):
        stageWhere=numpy.where(numpy.logical_and(stage==i,shift<.05))
#	stageWhere=numpy.where(shift<.1)
#	print stageWhere
        for j in stageWhere[0]:
                ax.loglog(fluxNorm[j],widthNorm[j],color=colors[i],marker=markerstyle[i],linestyle="")
        ax.plot(10,10, color=colors[i],marker=markerstyle[i],linestyle='',label="Stage "+str(i)) #plot pts for legend 
	#print "i=%d"%i
	#print len(colors)
    elif TIME_PLOT:
	for i in numpy.arange(len(byHistTimes)):
        	maxima=egan.findAzMax(Az[i,:],0,width,AVG=False,SHORTSTEP=True,DEBUG=False)
        	minima=egan.findAzMin(Az[i,:],0,width,AVG=False, SHORTSTEP=True)
        	#print minima
        	times=numpy.empty(len(maxima));times.fill(byHistTimes[i]*1E6)
        	times2=numpy.empty(len(minima));times2.fill(byHistTimes[i]*1E6)
        	ax.plot(minima,times2,linestyle='',marker='+',markersize=3,color='b')
        	ax.plot(maxima,times,linestyle='',marker='.',markersize=1.5,color='k')
	for i in stageTimes:
                #print i*1e6
                ax.plot([0,800],[i*1e6, i*1e6],color='k',linestyle='-')
        for i in range(1,max(dump)+1):
           dumpwhere=numpy.where(numpy.logical_and(dump==i,shift<.01))
	   #print "len xpos="
	   #print len(xpos)
           for j in dumpwhere[0]:
               	#print "i=%d,"%i
		#print " j=%d"%j
		ax.plot(xpos[j],time[j],color=colors[i],marker=markerstyle[i],linestyle='')
           ax.plot(1e4,1e4, color=colors[i],marker=markerstyle[i],linestyle='',label=str(i)) #plot pts for legend
    else:
        for i in range(1,max(dump)+1):
            dumpwhere=numpy.where(numpy.logical_and(dump==i,shift<.05))
	    for j in dumpwhere[0]:
                ax.loglog(fluxNorm[j],widthNorm[j],color=colors[i],marker=markerstyle[i],linestyle="")
            ax.plot(10,10, color=colors[i],marker=markerstyle[i],linestyle='',label=str(i)) #plot pts for legend later
    ax.set_title(title)
    plt.legend(loc=4,numpoints=1,fontsize='small',title="Step #")
    plt.draw()
    plt.savefig(plotDir+plotfile+".eps")
    plt.show()
    plt.close()




def dist_1d(width, \
                pathName,\
                resultsDir,\
                simLabel,\
                NX_TOTS,\
                DELTA,\
                LX,\
                B_0,\
                resultsDirs,\
                WIDTH=False,\
                FLUX=False,\
                SAVE_TOGETHER=False,\
		MULT_RUNS=False,\
 		STAGES=0): 
 ## For Stages, input the number of the stage that you want to plot. 0 for all stages
    print "1D Distribution Plotting"
    plotDir=resultsDir+"distplots/"
    if SAVE_TOGETHER:
        plotDir="041315_distplots/"
    runName="relRecon2p_"
    origfilename=pathName+runName+'yeeB_'+str(1)+".h5"
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
#    print filename
#    fluxNorm,widthNorm,xpos,time,stage=pplot.read_plasmoids(width,pathName,resultsDir,simLabel,NX_TOTS,DELTA, LX,B_0)

    ##########################
    ## Read in plasmoid info #
    ##########################
    widths=[]
    deltapsi=[]
    dump=[]
    xwidth=[]
    time=[]
    xpos=[]
    stage=[]
    if MULT_RUNS:
        for resultsDir in resultsDirs:
            for location in ["b","t"]:
                if location=="b":
                    filename=resultsDir+runName+"width_"+str(width)+"_extended.csv"
                else:
                        filename=resultsDir+runName+"width_"+str(width)+"T_extended.csv"
                #print location
                print filename
                with open(filename,'r') as csvfile:
                        freader=csv.reader(csvfile,dialect='excel')
                        head1=next(freader)
                        columns=next(freader)
                        for row in freader:
                                widths.append(float(row[11])/2+float(row[12])/2)
                                deltapsi.append(abs(float(row[4])-float(row[9])))
                                xwidth.append(abs(float(row[3])-float(row[10])))
                                dump.append(row[0])
                                xpos.append(row[3])
                                time.append(row[13])
				stage.append(row[14])
    ####################################
    # Convert arrays to proper form      ###
    ####################################         
    #print dump
    #execfile(pathName+"relReconVars.py") 
    dumpint= numpy.array(dump,dtype=int)#    print dumpint
    time=numpy.array(time,dtype=float)
    xpos=numpy.array(xpos,dtype=int)
    stage=numpy.array(stage, dtype=int)
    widthNorm=[abs(float(i))/NX_TOTS[0] for i in widths]
    fluxNorm=[abs(float(i))/(B_0*LX) for i in deltapsi]
    xwidthNorm=[abs(float(i)/NX_TOTS[0]) for i in xwidth]
    
    #print xwidthNorm
    if FLUX:
        plotfile="1Ddist_flux_"+str(1600)+"_width_"+str(width)+"allsims"
        title="1D Plasmoid Flux Distribution: All Sims, width="+str(width)
	xlabel="Normalized enclosed flux (deltaPsi/B_0*L)"
    else:
        title="1D Plasmoid Width Distribution:All Sims, width="+str(width)
        plotfile="1Ddist_width"+str(1600)+"_width_"+str(width)+"allsims"
	xlabel="Normalized y half-widths (wy/L)"
    ylabel="Frequency"
    if STAGES:
	plotfile=plotfile+"_stage"+str(STAGES)
	title="Stage "+str(STAGES)+" "+title
    bins=numpy.logspace(-5,0,num=30,base=10.0)
    #print bins
    left=13
    right=3

    if STAGES:
#	print STAGES
	stageWhere=numpy.where(stage==STAGES)
#	print "Flux norm start"
#	print fluxNorm
	placeHolder=[]
	for i in stageWhere[0]:
		placeHolder.append(fluxNorm[i])
	fluxNorm=placeHolder
#	print stageWhere


#	print "flux norm now?"
#	print fluxNorm
	
	
    if FLUX:
        hist,bin_edges=numpy.histogram(fluxNorm,bins=bins, density=1)
    else:
        hist,bin_edges=numpy.histogram(widthNorm,bins=bins, density=1)        
#  print len(fluxHist[0])
##############3
# Try linear regression
###############
    binfit=bin_edges[left:len(bin_edges)-(right+1)]
    histfit=hist[left:len(hist)-right]
    print histfit
    print hist
    #print fluxfit
    binfit=numpy.log10(binfit)
    histfit=numpy.log10(histfit)
    p=numpy.polyfit(binfit,histfit,1)
    print binfit
    print histfit
    print p[0]
    print p[1]
#print bin_edges
    #print bin_centers
    plt.loglog(bin_edges[0:len(bin_edges)-1],hist,linestyle='',marker='.')  
    x=bin_edges[left:len(bin_edges)-right-1]
    plt.loglog(x,10**p[1]*x**p[0])
    #plt.loglog(x,10**p[1]*x**-.8)    
    if STAGES:
    	plt.text(.0001,1,"Power Law: "+str(p[0]))
    else:
	plt.text(.0001,1,"Power Law: "+str(p[0]))
#plt.hist(fluxNorm,log=True,bins=bins,normed=1)	
    plt.gca().set_xscale("log")
    #plt.hist(widthNorm,bins=50)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(plotDir+plotfile+".eps")
    plt.show()
    plt.close()

"""
    ax = host_subplot(111, axes_class=AA.Axes)
    #y2=ax.twinx() #For 2nd axes
    ax.grid(True)

    ax.set_xlim(0,800)
    ax.set_ylim(0,max(time)*1.2)
   
    ax.set_yscale("log")


    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
"""

