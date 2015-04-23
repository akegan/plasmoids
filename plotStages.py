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
sys.path.append('/scr_verus/wernerg/vrun/relRecon/relReconRepo')
#import calcStats
import vorpalUtil
widths=[15]
#width=1
for runNum in [2,3,4]:
    print runNum
    width=15
    pathName="/scr_verus/wernerg/vrun/relRecon/plasmoids/m1-400-sig30-thd1p875-thbp01-res2-ppc3"+str(runNum)+"/"
    resultsDir="./size_1600_"+str(runNum)+'/'
    plotDir=resultsDir+"distplots/"
    runName="relRecon2p_"
    origfilename=pathName+runName+'yeeB_'+str(1)+".h5"
    MULTPLOTS=False
    HIST=False #2d histogram of wy vs psi
    WIDTHCOMP=False #wy on y wx on x
    AREA=False #wx*wy vs psi
    NORMAL=True
    DEBUG=False
    STAGES=True

#Read in plasmoid info
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
    execfile(pathName+"relReconVars.py")
    print NX_TOTS[0]
    filename=resultsDir+runName+"width_"+str(width)+"_extended.csv"
    print filename
    with open(filename,'r') as csvfile:
        freader=csv.reader(csvfile,dialect='excel')
        head1=next(freader)
        #head2=next(freader)
        columns=next(freader)
        widths=[]
        deltapsi=[]
        dump=[]
        xwidth=[]
	time=[]
        #print columns
        #print columns[11]
        for row in freader:
            widths.append(float(row[11])/2+float(row[12])/2)
            deltapsi.append(abs(float(row[4])-float(row[9])))
            xwidth.append(abs(float(row[3])-float(row[10])))
            dump.append(row[0])
            time.append(row[13])
		#print row
    ##NOW FOR TOP
    filename=resultsDir+runName+"width_"+str(width)+"T_extended.csv"
    print filename
    with open(filename,'r') as csvfile:
        freader=csv.reader(csvfile,dialect='excel')
        head1=next(freader)
        columns=next(freader)
        #print columns
        #print columns[11]
        for row in freader:
            #print row
            widths.append(float(row[11])/2+float(row[12])/2)
            deltapsi.append(abs(float(row[4])-float(row[9])))
            xwidth.append(abs(float(row[3])-float(row[10])))
            dump.append(row[0])
	    time.append(row[13])
    #Get B_0 and system + cell size
    execfile(pathName+"relReconVars.py")
    print NX_TOTS[0]

    ##Dividing width in cell size by total number of cells should be enough 
    ##wx/L=widths/400 (not 401 because the extra 1 does not count in system size)
    widthNorm=[abs(float(i))/NX_TOTS[0] for i in widths]
    fluxNorm=[abs(float(i))/(B_0*LX) for i in deltapsi]
    xwidthNorm=[abs(float(i)/NX_TOTS[0]) for i in xwidth]
    dumpint= numpy.array(dump,dtype=int)
    time=numpy.array(time,dtype=float)


    colors=["b","g","r","c","m","y"]
    markerstyle=['.','o','v','^','<','>','8','s','p','*','+','x','d','D']
    markerstyle=markerstyle*5
    colors=colors*len(dump)

    #######################
    ## SET UP PLOT    #####
    #######################
    ax = host_subplot(111, axes_class=AA.Axes)
    y2=ax.twinx() #For 2nd axes
    ax.grid(true)
    ax.set_xlim(10**-5,1)
    ax.set_ylim(10**-5,1)
    plotfile="plasmoidDist_"+str(width)
    ylabel="Normalized y widths (wy/L)" #default val
    xlabel="Normalized enclosed flux (deltaPsi/B_0*L)" #default val

    ax.loglog([10**-5,1],[10**-5,1],linestyle="-",color='k',linewidth=1) #plot 1:1 line
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    x=numpy.arange(1,10**5,10**2)
    x=x*(10**-5)
    #SET RIGHT AXIS
    if AREA==False:
        y2.set_ylabel("Number of Cells")
        y2.set_yscale("log")
        y2.set_ylim(10**(-5)*NX_TOTS[0],1*NX_TOTS[0])
        y2.axis["right"].label.set_color('k')
    if NORMAL or HIST: #Plot lines
        title="Distribution of Plasmoid fluxes wrt Y-width, PeakWidth="+str(width)
        ax.loglog([.000005,1],[NX_TOTS[0]**-1,NX_TOTS[0]**-1],linestyle='--',color='k') ##Plot line of cell siz
        ax.loglog([.000005,1],[DELTA/LX,DELTA/LX],linestyle='-',color='k') ##Plot line of reconnection layer width
        ax.loglog(B_0*DELTA*numpy.log(numpy.cosh(x*NX_TOTS[0]/DELTA)),x,linestyle='-')
#####################
## Plot the values ##
#####################
######
# Get the stages ##
###################

    simName=pathName+"relRecon2p"
    fluxFn=calcStats.getUnreconnectedFluxVsTime(simName)
    dnByName = "layerDnByLine"
    byHistTimes = vorpalUtil.getHistoryTimes(simName, dnByName)[:,0]
    returns=calcStats.fit2reconRate(byHistTimes, fluxFn(byHistTimes))
    coords=returns[2]
    stageTimes=1e6*coords[:,0]
##get times
    numStages=len(stageTimes)
    print stageTimes
    print time
 
    for i in range(0,len(stageTimes)-1):
	if i==0:
		stageWhere=numpy.where(time<stageTimes[i])
	else:
		stageWhere=numpy.where(numpy.logical_and(time<stageTimes[i],time>stageTimes[i-1]))
	for j in stageWhere[0]:
		ax.loglog(fluxNorm[j],widthNorm[j],color=colors[i],marker=markerstyle[i],linestyle="")
        ax.plot(10,10, color=colors[i],marker=markerstyle[i],linestyle='',label="Stage "+str(i)) #plot pts for legend later
	
    ax.set_title(title)
    plt.legend(loc=4,numpoints=1,fontsize='small',title="Step #")
    plt.draw()
    plt.savefig(plotfile+"_bystages.png",dpi=500)
    plt.show()
    plt.close()

