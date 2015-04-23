import sys
sys.path.append('/scr_verus/wernerg/vrun/relRecon/relReconRepo')

import calcStats
import vorpalUtil
import numpy
import tables
import scipy
import matplotlib
import matplotlib.pyplot as plt
import collections
import scipy.signal
import csv
import egan_vorpalUtil as egan
simNum=2
simName="/scr_verus/wernerg/vrun/relRecon/plasmoids/m1-400-sig30-thd1p875-thbp01-res2-ppc3"+str(simNum)+"/relRecon2p"
fluxFn=calcStats.getUnreconnectedFluxVsTime(simName)
dnByName = "layerDnByLine"
byHistTimes = vorpalUtil.getHistoryTimes(simName, dnByName)[:,0]
#print byHistTimes
print len(byHistTimes)
print "something else"
print fluxFn(4002)
#print fluxFn(byHistTimes)

returns=calcStats.fit2reconRate(byHistTimes, fluxFn(byHistTimes))
print returns[3]
coords=returns[2]
reconStageTimes=coords[:,0]
print reconStageTimes
finalFn=returns[3]
plt.plot(byHistTimes*1e6,finalFn(byHistTimes),linestyle='-',color='b')
plt.plot(reconStageTimes*1e6,fluxFn(reconStageTimes),linestyle='',color='r',marker='o')
plt.plot(byHistTimes*1e6,fluxFn(byHistTimes),linestyle='--',color='k')
plt.title("Reconnection Stages Based on Unreconnected Flux")
plt.xlabel("Time (1e-6 seconds)")
plt.ylabel("Unreconnected Flux")
plt.savefig("ReconnectionStages.eps")
plt.show()


