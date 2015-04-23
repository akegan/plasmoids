import numpy
import tables
import scipy
import matplotlib.pyplot as plt
import matplotlib
import collections
import scipy.signal
import csv


def makeAz(filename): #returns Az
    nodeName='yeeB'
    file=tables.openFile(filename)
    b=file.getNode('/'+nodeName)
    gridinfo=file.getNode('/'+'compGridGlobal')
    #determine grid cell dimensions
    upbd=gridinfo._v_attrs.vsUpperBounds
    lwbd=gridinfo._v_attrs.vsLowerBounds
    numCells=gridinfo._v_attrs.vsNumCells
    dx=(upbd[0]-lwbd[0])/numCells[0]
    dy=(upbd[1]-lwbd[1])/numCells[1]
    #generate Az from b
    bx=b[:,:,0]
    by=b[:,:,1]
    nx=b.shape[0]
    ny=b.shape[1]
    Az=numpy.zeros((nx+1,ny+1))
    #First integrate from (0,0) to (0,y) to find Az(0,y) for all y
    Az[0,1:ny+1]=bx[0,:ny].cumsum()*dy
    #now integrate from Az(0,y) to Az(x,y) **note dx->-dx
    Az[1:nx+1,:ny]=Az[0,:ny]+by[:nx,:ny].cumsum(axis=0)*-dx
    #add the top row, assuming rotationless field, pulled directly from lineintegrate2d
    xRaTop = bx[1:nx, ny-1] - bx[0:nx-1, ny-1] + by[0:nx-1, ny-1]
    Az[1:nx,ny] = Az[0,ny] + xRaTop[:nx-1].cumsum() * -dx
    file.close() #close the h5 file
    return Az
    
def plotAzSlice(Az,ypos,title):
    #title='Az at cellnum=100, data dump 5'
    xlabel='x position (cell number)'
    ylabel='Az'
    plt.plot(Az[:,ypos])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def findAzMin(Az,ypos,width,AVG=False,SHORTSTEP=False): #width defines how many cells on either side
    if AVG:
        AzSlice=numpy.add(Az[:,ypos],Az[:,ypos+1])
        numpy.add(AzSlice,Az[:,ypos-1],AzSlice)
        AzSlice=AzSlice/3
    elif SHORTSTEP:
        AzSlice=Az
    else:
        AzSlice=Az[:,ypos]
    minima=scipy.signal.argrelextrema(AzSlice, numpy.less,order=width, mode='wrap')
    minima=minima[0]
    return minima
    
def findAzMax(Az,ypos,width,AVG=False,SHORTSTEP=False,DEBUG=False): #w
    if AVG:
        AzSlice=numpy.add(Az[:,ypos],Az[:,ypos+1])
        numpy.add(AzSlice,Az[:,ypos-1],AzSlice)
        AzSlice=AzSlice/3
    elif SHORTSTEP:
        AzSlice=Az[:]
    else:
        AzSlice=Az[:,ypos]
 
    maxima=scipy.signal.argrelextrema(AzSlice, numpy.greater,order=width, mode='wrap')
    if DEBUG:
        print "Az slice shape"
        print AzSlice.shape
        #print AzSlice
        print "all maxima?"
        print maxima
    maxima=maxima[0]
   # print "Az Slice:"
   # print AzSlice.shape
   # print len(Az[:,ypos])
   # print maxima
    return maxima

def plotAzMaxMin(Az, pArray,ypos, maxima, minima,dump,savefilename): #Updated to plot found x and o pts. 
    ylabel="Az"
    xlabel="X Position (Cell Number)"
    title="X and O Points on Linescan of Az. DumpNum="+str(dump)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    plt.plot(Az[:,ypos])
    plt.plot(maxima,Az[maxima,ypos],marker='.',linestyle='')
    plt.plot(minima,Az[minima,ypos],marker='.',linestyle='')
    num=len(pArray)-1
    colors=["b","g","r","c","m","y"]
    colors=colors*num
    for i in range(1,num+1):
        plt.plot(pArray[i].xloc,Az[pArray[i].xloc,ypos],marker='o',linestyle='',color=colors[i-1])
        plt.plot(pArray[i].xmin,Az[pArray[i].xmin,ypos],marker=(5,1),linestyle='',color=colors[i-1])

    plt.savefig(savefilename)
  #  plt.show()
    plt.clf()


def interpolate(Az, ypos, index): #Finds more precise MAX, indices can be max or mins
    x_step=1 #for now, keep lengths in unit cells)
    f_prime=(Az[index+1,ypos]-Az[index-1,ypos])/(2*x_step)
    f_2prime=(Az[index+1,ypos]-2*Az[index,ypos]+Az[index-1,ypos])/(x_step)**2
        
    dx=-f_prime/f_2prime #distance to add to initial distance
    f_interp=Az[index,ypos]+dx*.5*f_prime+(dx)**2*f_2prime
    return dx, f_interp
  


def plasmoids_2nd(ypos,\
		Az,\
		maxima,\
		minima,
		runName,\
		dump,\
                time,\
                heightParam=.5,\
                DEBUG=False): 
#gives 1st order info we want about plasmoids in x, 2nd order ytop, ybot
    #Define structures (named tuples) to store data in
    #plasmoidnum =>which plasmoid in this dump
    #yloc -> y cell number of scan
    #xloc -> x cell number of maximum flux
    #phimax ->flux value at maximum
    #xL, xR-> x cell numbers of neighboring minima
    #phiL, phiR -> value of flux at minima
    #chosen minimum flux (larger of the two, to start)
    #ytop, ybot -> width in top and bottom direction 
    data_headers=["stepnum",\
		"plasmoidnum", \
		"ypos", \
		"xloc", \
		"phimax", \
		"xL", \
		"xR", \
		"phiL", \
		"phiR", \
		"phimin", \
                "xmin",\
		"ytop", \
		"ybot",\
                "time"]
    dataStruct=collections.namedtuple("dataStruct",data_headers)
    pArray=[0]
    x_step=1
    i=0

    #***Maybe add in wrapping neighboring minimum location later
    while i<len(maxima):

    #FIND MINIMA SURROUNDING MAX
        if DEBUG:
            print "Max number i= %d (top of loop)" %(i)
        SWITCH=True #turn switch on, ie we still want to find xRight and Left
        if maxima[i]<minima[0]: #if we start with a max, look to the RHS
            xLeft=minima[len(minima)-1]#if we start with a max, then the left min is the last min
            xRight=minima[0]
            SWITCH=False #mark that we no longer need to find xleft or x right
            if DEBUG:
                print "Have edge plasmoid on left"
        if maxima[i]>minima[len(minima)-1]: #if it ends with a maximum, and we get there, we're done
            xRight=minima[0]#if we end with a max, then the min is the first one on the left
            xLeft=minima[len(minima)-1]
            SWITCH=False
            if DEBUG:
                print "Have edge plasmoid on right"
        if SWITCH: #if not edge plasmoid, 
            for j in xrange(len(minima)):
                if DEBUG: 
                   print "Checking which min is xR, j= %d" %(j)
                if maxima[i]<minima[j]:
                    xRight=minima[j]
                    xLeft=minima[j-1]
                    break #break for loop
        if DEBUG:
            print "xR= %d, xL=%d" % (xRight,xLeft)

    #CHOOSE WHICH MIN TO USE AS REFERENCE
            #Choose the biggest of the 2, UNLESS, heightParam switch is set.
        chosenPhi=max(Az[xLeft,ypos],Az[xRight,ypos])#choose the bigger of the 2 for the reference minimum

        if Az[xLeft,ypos]==chosenPhi:
            xmin=xLeft
            xother=xRight
        else:
            xmin=xRight
            xother=xLeft
        if Az[maxima[i],ypos]<chosenPhi: #If the min is greater than the max
            chosenPhi=Az[xother,ypos]
            hold=xother
            xother=xmin
            xmin=hold
        if 0==1:
            if (Az[maxima[i],ypos]-Az[xmin,ypos])<heightParam*(Az[maxima[i],ypos]-Az[xother,ypos]):
                    x2=xother #just to store number as I switch
                    xmin=xother
                    xother=x2
                    chosenPhi=Az[xmin,ypos]
                    print "Hit the height Param switch"
                    #if DEBUG:
                    print "xmin: %d"%(xmin)
                    print "Chosenphi %f"%(chosenPhi)
                    print "other phi %f"%(Az[xother,ypos])
    #FIND TOP EDGE
        TOO_BIG=False #Don't count plasmoid unless it fits in x/y
        step=ypos
        while Az[maxima[i],step]>chosenPhi:
            step +=1
            if step==len(Az)-1:
                if DEBUG:
                    print "Plasmoid is too big (top edge)"
                TOO_BIG=True
                break
        ytop=step
    #FIND BOTTOM EDGE
        step=ypos
        while Az[maxima[i],step]>chosenPhi:
            step-=1
            if step==0:
                if DEBUG:
                    print "Plasmoid is too big (bottom edge)"
                TOO_BIG=True
                break
        ybot=step
        if TOO_BIG:
            if DEBUG:
                print "Breaking Plasmoid loop because plasmoid is too big"
            break

    #CORRECT YTOP, YBOT TO 1ST ORDER (approximate as line)
        #YTOP
        dAz=chosenPhi-Az[maxima[i],ytop] #difference between current and desired
        slope=Az[maxima[i],ytop]-Az[maxima[i],ytop-1]/x_step #This is negative
        dy=dAz/slope #This is negative
        ytop=ytop+dy
        if DEBUG:
            print "Width Checking..."
            print "dytop=%f" %(dy)
            #print "slope top=%f" %(slope)
        #YBOT (with some added negative signs
        dAz=chosenPhi-Az[maxima[i],ybot]
        slope=Az[maxima[i],ybot+1]-Az[maxima[i],ybot]/x_step #This is positive
        dy=dAz/slope #positive
        ybot=ybot+dy #Brings closer to peak
        if DEBUG:
            print "dybot=%f" %(dy)
        #STORE DATA IN STRUCT AND ADD TO ARRAY
        plasmoid=dataStruct(stepnum=dump,\
			plasmoidnum=i,\
			ypos=ypos, \
			xloc=maxima[i], \
			phimax=Az[maxima[i],ypos],\
			xL=xLeft, \
			xR=xRight,\
			phiL=Az[xLeft,ypos],\
			phiR=Az[xRight,ypos],\
			phimin=chosenPhi,\
                        xmin=xmin,\
			ytop=ytop-ypos,\
			ybot=ypos-ybot,\
                        time=time)
        pArray.append(plasmoid)
        i+=1
    
    return pArray #namedtupple of quantities at top

def writeExtended(pArray,filename,dump): #Write all of plasmoid parameters to file
    with open(filename,"a") as csvfile:
        awriter=csv.writer(csvfile,dialect='excel')
        for i in range(1,len(pArray)):
            awriter.writerow(pArray[i])
    csvfile.close()

def writeShort(pArray,filename,dump):
    #COLUMNS: DumpNum, PlasmoidNum, AvgWidth, DeltaPsi
     with open(filename,"a") as csvfile:
        awriter=csv.writer(csvfile,dialect="excel")
        for i in range(1,len(pArray)):
             avgwidth=(pArray[i].ytop+pArray[i].ybot)/2
             deltapsi=pArray[i].phimax-pArray[i].phimin
             row=[dump,pArray[i].plasmoidnum,avgwidth,deltapsi]
	     awriter.writerow(row)
     csvfile.close()
    
def startShort(pathName,runName): #Starts csv with width, deltapsi, runName is identifier
    filename=pathName+runName+'_short.csv'
    file_header=["run/data name="+runName]
    file_header2=["Avg Width is the average of top and bottom widths, deltapsi is phimax-phimin (chosen based on highest neigboring min)"]
    data_headers=['StepNum','plasmoidnum','avgwidth','deltapsi']
    with open(filename,"w") as csvfile:
        awriter=csv.writer(csvfile,dialect="excel")
        awriter.writerow(file_header)
        awriter.writerow(file_header2)
        awriter.writerow(data_headers)
    csvfile.close()
    return filename
    
def startExtended(pathName,runName,data_headers): #starts csv file with all stored in pArray, runName is just identifier
    filename=pathName+runName+'_extended.csv'
    file_header=["run/data name="+runName]
    with open(filename,"w") as csvfile:
        awriter=csv.writer(csvfile, dialect='excel')
        awriter.writerow(file_header)
        awriter.writerow(data_headers)
    csvfile.close()
    return filename


def plotPlasmoidContoursWTop(ypos,\
                             yposT,\
                             Az,\
                             maxima,\
                             minima,\
                             maxT,\
                             minT,\
                             pArray,\
                             pArrayT,\
                             dump,\
                             outName,\
                             nLevels=40,\
                             DEBUG=False,\
                             ZOOM_SWITCH=False,\
                             GRID=False,\
                             DPI=1000):
    #START WITH BASE CONTOURS   
    x=range(0,len(Az))
    xlabel="x (direction of reconnecting sheet) in cell num"
    ylabel="y (perp to reconnecting sheet) in cell num"
    plt.title("Plot of Calculated Plasmoids, DumpNum="+str(dump))
    AzT=Az.T #Plot transpose of Az so that it matches Greg's
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #get level spacing right (match greg's)
    dataMax=abs(Az).max().max()
    levelSpace=2*dataMax/nLevels
    levelMax=numpy.ceil(dataMax/levelSpace)*levelSpace
    #plot base contour
    if ZOOM_SWITCH:
        ybot=ypos-len(Az)/16
        ytop=ypos+len(Az)/16
        cs=plt.contour(x,x[ybot:ytop],AzT[:,ybot:ytop],
                levels=numpy.arange(-levelMax,levelMax,levelSpace),\
                norm=matplotlib.colors.Normalize(vmin=-levelMax,vmax=levelMax))
    else:
        cs=plt.contour(x,x,AzT,\
                levels=numpy.arange(-levelMax,levelMax,levelSpace),\
                norm=matplotlib.colors.Normalize(vmin=-levelMax,vmax=levelMax))
    cbar=plt.colorbar(cs,extend='neither')
    if GRID:
        plt.minorticks_on()
        plt.grid(b=True,which='both',color='.9',linestyle='-')

    #PLOT PLASMOID OUTLINE + X and O pts(or at least attempt to)
    for location in ["bot","top"]:
        if ZOOM_SWITCH and location=="top":
            if DEBUG:
                print "Breaking due to zoom_switch"
            break
        if DEBUG:
            print "Top or Bottom? %s"%(location)
        if location=="top":
            pArray=pArrayT
            ypos=yposT
        num=len(pArray)-1
        ymin=int(ypos-len(Az)/4)
        ymax=int(ypos+len(Az)/4)
        if DEBUG:
            print "Number of plasmoids to plot: %d"%(num)
        delta=.0001
        colors=["b","g","r","c","m","y"]
        colors=colors*num
        for i in range(1,num+1):
            if DEBUG:    
                print "On plasmoid number: i=%d"%(i)
            #plot contour outline (this actually plots 3 contours, but very close together)
            if location=="top":
                levels=numpy.arange(-pArray[i].phimin-delta,-pArray[i].phimin+delta,delta)
            else:
                levels=numpy.arange(pArray[i].phimin-delta,pArray[i].phimin+delta,delta)

        #Determine how far on either side to plot contour
            plus=1 #How much to add on either side(just to see)
            if pArray[i].xL>pArray[i].xR: #If we have an either edge edge plasmoid
                if DEBUG:
                    print "Potential Edge Plasmoid Problems. xR=%d, xL=%d"%(pArray[i].xR,pArray[i].xL)
                x2max=pArray[i].xR+plus
                x2=numpy.arange(0,x2max)
                x3min=pArray[i].xL-plus
                x2=numpy.append(x2,numpy.arange(x3min,len(Az))) #Add on the RHS
                Acolumns=numpy.vsplit(AzT,[ymin,ymax])
                AzYsplit=Acolumns[1]#gives us just the columns that we want
                Arows=numpy.hsplit(AzYsplit,[x2max,x3min])
                AzT2=numpy.hstack((Arows[0],Arows[2]))
                if DEBUG:
                    print "AzT Shape:"
                    print AzT.shape
                    print "New Shape:"
                    print AzT2.shape
                    print "ydim= %d, xdim=%d"%(len(range(ymin,ymax)),len(x2))
            else:   
                x2min=pArray[i].xL-plus
                x2max=pArray[i].xR+plus
		#Watch out for plasmoids too close to the edge!
		if x2min<0:
			x2min=pArray[i].xL
		if x2max>len(Az):
			x2max=pArray[i].xR
		#print x2min
		#print x2max
		x2=range(x2min,x2max)
                AzT2=AzT[ymin:ymax,x2min:x2max] ##finish updating this, can i add 2 arrays together?
            if DEBUG:
                print "X-range for plasmoid:"
                print x2

            y2=range(ymin,ymax)
            if DEBUG:
                print "Checking to see if I have to reduce AzT to make this work"
                print "Moved onto next step of debugging!"
                print "len x2=%d, len y2=%d"%(len(x2),len(y2))
                print "Shape of AzT2:"
                print AzT2.shape
            plt.contour(x2,y2,AzT2,\
                    levels,
                    colors='k',\
                    linewidths=1,\
                    linestyles="-")
            #plot O and x pt, width on each side
            plt.plot(pArray[i].xloc,ypos,marker='o',linestyle='',color=colors[i-1])
            plt.plot(pArray[i].xmin,ypos,marker=(5,1),linestyle='',color=colors[i-1])
            ##Try to Plot the outline of one cell.
            plt.plot([100,101],[100,100],linestyle='-',color='k')
            plt.plot([100,100],[100,101],linestyle='-',color='k')
            plt.plot([101,101],[100,101],linestyle='-',color='k')
            plt.plot([100,101],[101,101],linestyle='-',color='k')
            
    plt.savefig(outName)
    #plt.show()
    plt.clf()

def getTime(filename):
    nodeName='time'
    file=tables.open_file(filename)
    times=file.get_node('/'+nodeName)
    time=times._v_attrs.vsTime
    time6=time*1E6
    #Other option is vsStep
    file.close()
    return time6
    
    #determine grid cell dimensions
    #upbd=gridinfo._v_attrs.vsUpperBounds
    #lwbd=gridinfo._v_attrs.vsLowerBounds
    
    


