import itertools
import re
import mathPlus
import optPlus
import arrayPlus
import genUtil
import vorpalUtil
import numpy
import numpy.linalg
import scipy
import scipy.optimize
import scipy.special
import scipy.interpolate
import scipy.integrate
import string
#import disthistMac

# for debugging
#import warnings

c=2.99792458e8
mu0=4e-7*numpy.pi

suppressOutput = False

# default max function evaluations for curve fitting optimization
defMaxFev = 50000
defFontSize = 22

def setFontSize():
  import matplotlib
  matplotlib.rcParams.update({'font.size': defFontSize,
                            'axes.labelsize':defFontSize+2})
  return

def getEnergyDisFrac(sigma, Ly, rho0): #{
  """ for square sims"""
  if sigma >= 9.9:
    res = 60.8 - 897./(0.5*Ly/rho0 * 16./sigma)
  elif abs(sigma-3.) < 0.1:
    res = 60.8 - 160./(0.5*Ly/rho0 * 16./sigma)**0.5
  else:
    msg = "cannot handle sigma = %g" % sigma
    raise ValueError, msg
  return res/100.
#}

def subtractMaxwell(gEdges, dNdg, thb, m = None): #{
  """
  Fits a maxwellian to the low-energy part of dNdg vs g
  where g is gamma.
  maxwellian(g) = A g sqrt{g^2-1} exp(-(g-1)/thb)
  gEdges = array of length N+1
  dNdg = array of length N, where dNdg[i] contains the number of 
    particles with gamma between gEdges[i] and gEdges[i+1]
  thb = the initial (or expected) temperature of particles
  m = integer method of fitting the maxwellian
    0: assumes maxwellian of temperature thb, and simply sets A
      to fit dNdg[0]
    1: don't use this
    2: From the temperature thb, 
        computes an average gamma for the associated Maxwellian;
       Finds maximum value of dNdg up to the average gamma, and
       fits a Maxwellian (for A and new temp thb) to dNdg up to
       the gamma where that max of dNdg occurs.
    3: Iterates 2 until thb converges.
  returns (dNdgSub, A, thb, fail)
    dNdgMaxwell = the fitted Maxwellian (same shape as dNdg)
    A = fitted value of A (see maxwellian(g))
    thb = fitted value of thb (see maxwellian(g))
    fracInMax = the fraction of particles in the low-energy maxwellian
    energyFracInMax = the fraction of (kinetic) energy in the low-energy maxwellian
      e.g., not gamma but (gamma-1)
    gMinInd: gEdge[gMinInd] is the smallest gamma for which 
      dNdgSub should be trusted, due to negative values in dNdgSub
    fail = 0 if success; if m=2 or 3 and there's an error (e.g., failure
      to converge), then returns value for m=0 with fail=1.

  N.B. to subtract the maxwellian, I recommend taking
    abs(dNdg - dNdgMaxwell) to avoid negative numbers.
  """
  dg = gEdges[1:] - gEdges[:-1]
  g = 0.5 * (gEdges[1:] + gEdges[:-1])
  fail = 0
  def maxwell(g, A, th): #{
    #print "maxwell called with A, th =", repr(A), repr(th)
    res = A * g * numpy.sqrt((g+1.)*(g-1.))*numpy.exp(-(g-1.)/abs(th))
    return res
  #}
  def intMaxwell(ge, A, th): #{
    """ integrated maxwell: ge = edges
    """
    #print "intMaxwell called with A, th =", repr(A), repr(th)
    res = 0.*ge[1:]
    for i in range(ge.size-1):
      res[i] = scipy.integrate.quad(maxwell, ge[i], ge[i+1], args=(A,th))[0]
      res[i] /= ge[i+1]-ge[i]
    return res
  #}
  if m is None:
    if (dNdg > 0.).sum() <= 2:
      m = 0
    else:
      m = 3
  if m == 0: #}{
    A = dNdg[0]/maxwell(g[0], 1., thb)
  elif m == 1: #}{
    gAvg = (g * dNdg * dg).sum() / (dNdg * dg).sum()
    thb = gAvg/3.
    A = dNdg[0]/maxwell(g[0], 1., thb)
  elif m == 2 or m == 3: #}{
    thbOld = thb
    def maxwell2(g, A, th): #{
      return numpy.log10(A * g * numpy.sqrt(g**2-1.)*numpy.exp(-g/th))
    #}
    thbs = [thb]
    # not clear if scaleFactors do any good (or if I'm using them correctly)
    # N.B. do not let these get negative
    scaleFactors = None #[numpy.log(dNdg[0]/maxwell(g[0],1.,thb)), thb]
    converged = False
    iteration = 0
    try: #{
      while not converged: #{
        # can't evaluate k1 for very small thb
        # gb = scipy.special.k1(1./thb)/scipy.special.kn(2,1./thb)+3*thb
        gb = 1. + disthistMac.avgMaxwellianGammaM1(thb)
        if g[1] > gb:
          ptslgb = 4
        elif gb >= g[-10]:
          ptslgb = g.size - 10
        else:
          ptslgb = numpy.where(g>=gb)[0][0] + 4
        igmax = numpy.argmax(dNdg[:ptslgb])
        pts = max(7, igmax)
        #print iteration, gb, ptslgb, pts, g[ptslgb], g[igmax]
        #pts = numpy.where(g>=gb)[0][0] + 3
        #print pts
        if 0: #{
          import pylab
          pylab.semilogx(g, dNdg)
          pylab.semilogx(g[:pts], dNdg[:pts], 'o', mfc='none')
          pylab.show()
          import sys
          sys.exit()
        #}
        #logdNdg = numpy.log10(dNdg[:pts])
        A0 = dNdg[0] / maxwell(g[0], 1., thb)
        thb0 = max(g[0]-1, thb)
        if 0: #{
          import pylab
          pylab.loglog(g, intMaxwell(gEdges,A0, thb))
          pylab.loglog(g, dNdg, 'o', mfc='none')
          pylab.loglog(g[igmax], dNdg[igmax], 'x')
          pylab.show()
        #}
        #print A0, thb0, scaleFactors
        #print "Starting opt"
        #(popt,pcov) = scipy.optimize.curve_fit(
        #  maxwell, g[:pts], dNdg[:pts], p0=(A0,thb0), maxfev=60000,
        #  diag = scaleFactors)
        #print "About to fit for maxwellian starting at A0, thb0", A0, thb0
        #print "   using", pts, "points"
        (popt,pcov) = scipy.optimize.curve_fit(
          intMaxwell, gEdges[:pts+1], dNdg[:pts], p0=(A0,thb0), maxfev=60000,
          diag = scaleFactors)
        #(popt,pcov) = scipy.optimize.curve_fit(
        #  maxwell2, g[:pts], logdNdg, p0=(dNdg[0],1.))
        A = popt[0]
        thbLast = thb
        thb = abs(popt[1])
        #print "iteration, A, thb, dif", iteration, A, thb, thb-thbLast
        thbs.append(thb)
        fit = maxwell(g[:pts], A, thb)
        if m == 2 or abs(1.-thb/thbs[-2]) < 0.01:
          converged = True
        iteration += 1
        if iteration > 1000:
          raise ValueError, "iteration for maxwellian fit exceeded " + str(iteration-1)
      #}
    except: #}{
      raise
      fail = 1
      #raise
      #return subtractMaxwell(gEdges, dNdg, thbOld, m=0)
      #print g[0], dNdg[0], thbOld, maxwell(g[0],1.,thbOld)
      A = dNdg[0]/maxwell(g[0], 1., thbOld)
      #print A
    #}
    #print thbs, gb, pts
  else: #}{
    raise ValueError, "m = %i is invalid" % m
  #}
  # calculate maxwellian with thb
  #print "final", A, thb
  dNdgMax = maxwell(g, A, thb)
  dNdgSub = dNdg - dNdgMax
  if 0:
    import pylab
    pylab.semilogx(g, dNdgSub)
    pylab.show()
  # find small g such that dNdgMax < dNdg for all larger g
  z = numpy.logical_and(dNdgSub <= 0, dNdg != 0)
  if z.any():
    gMinInd = numpy.where(z)[0][-1]+1
    if gMinInd >= z.size:
      gMinInd = z.size-1
  else:
    gMinInd = 0
  # Don't return gMinInd if dNdg[gMinInd] = 0
  while (gMinInd > 0) and dNdg[gMinInd] == 0.:
    gMinInd -= 1
  # calculate fraction of particles in low-energy maxwellian
  if 1: #{
    dg = gEdges[1:] - gEdges[:-1]
    dN = dNdg * dg
    dNMax = dNdgMax * dg
    maxTooMuch = (dNMax > dN)
    dNMax[maxTooMuch] = dN[maxTooMuch]
    dNsum = dN.sum()
    if dNsum == 0.:
      fracInMax = 0.
    else:
      fracInMax = dNMax.sum() / dN.sum()
    energyFracInMax = (dNMax * (g-1.)).sum() / (dN * (g-1.)).sum()
  #}
  # and subtract
  if 1: #{
    dNdgSub = abs(dNdgSub)
  else: #}{
    maxRed = 1e-4
    overRed = (dNdgSub < 0)
    dNdgSub[overRed] = 0.01*maxRed*dNdg[overRed]
    overRed = numpy.logical_and(dNdgSub < maxRed*dNdg, numpy.logical_not(overRed))
    dNdgSub[overRed] = maxRed * dNdg[overRed]
  #}
  #print "Using", pts, "points"
  #import os
  #print os.getcwd()
  return (dNdgMax, A, thb, fracInMax, energyFracInMax, gMinInd, fail)
#}

def saturationPoint(ra, fracOfSaturation, logScale = False): #{
  """
  supposes that ra represents an increasing sequence; ideally, but not
  actually monotonic.

  returns (i1, i2) where i1 is the smallest index where ra first goes above
  the maximum value times fracOfSaturation, and i2 is the highest index where
  ra is below that.
  """
  iMin = numpy.argmin(ra)
  raMin = ra[iMin]
  iMax = iMin + numpy.argmax(ra[iMin:])
  raMax = ra[iMax]
  def growth(x, A, a, x0): #{
    res = A*(1.-numpy.exp(-a*(x-x0)))
    return res
  #}
  xs = numpy.arange(len(ra))
  A0 = raMax
  a0 = 10./xs[-1]
  x00 = 0.
  (popt, pcov) = scipy.optimize.curve_fit(growth, xs[iMin:], ra[iMin:], p0 = (A0, a0, x00))
  (A, a, x0) = popt
  nra = growth(xs, *popt)
  nraMax = min(A, raMax)
  if logScale:
    threshold = numpy.exp(
      numpy.log(raMin) 
      + fracOfSaturation*(numpy.log(nraMax) - numpy.log(raMin)))
  else:
    threshold = raMin + fracOfSaturation*(nraMax - raMin)
  graph = False
  try:
    i1 = iMin + numpy.where(nra[iMin:] >= threshold)[0][0]
    i2 = iMin + numpy.where(nra[iMin:] <= threshold)[0][-1]
  except:
    import sys
    i1 = 0
    i2 = 1
    msg = os.getcwd() + ": Failed to find g2 saturation point\n"
    sys.stderr.write(msg)
    raise
    #"iMin, iMax = ", iMin, iMax
    #print "nra.size = ", nra.size
    #print "Finding i1, i2 failed: setting to 0 and 1"
    #graph = True
  if graph: #{
    import pylab
    print "ra: %g -> %g" % (raMin, raMax)
    print "  frac: %g" % fracOfSaturation
    print "  threshold: %g" % threshold
    print "  i1, i2 = %i, %i" % (i1,i2)
    print "  popt =", popt
    xs = numpy.arange(len(ra))
    ys = growth(xs, *popt)
    pylab.semilogy(xs, ra, ':')
    pylab.semilogy(xs, ys, '-')
    pylab.semilogy(i1, ra[i1], 'o', mfc='none')
    pylab.semilogy(i2, ra[i2], 's', mfc='none')
    pylab.show()
  #}
  return (i1, i2)
#}

def fit1(gEdges, dNdg, thb): #{
  """
  fits 
  dNdg = C g^a e^{b1 g + b2 g^2 + c1/(g-1)} 
  returns C, a, b1, b2, c1, cov
   where cov is the normalized covariance of the parameters
   let (C, a, b) = (a_0, a_1, a_2)
   Then cov_ij = < [a_i - <a_i>] [a_j - <a_j>] >
   where <a_i> is the value returned (e.g., C, a, or b).
   E.g., sqrt(cov_ii) is the std. dev. of a_i.

   If the ySigmas are known, then the "real" cov can be find by
   real cov = N/(N-M) r^2 cov
   where r^2 = Sum_n (y_n - fitted_y_n(C,a,b))^2 / ySigmas_n^2.
  """
  def lnPowLawExp2(g,lnC,a,b1,b2,c1):
    res = lnC + a*numpy.log(g) + b1*g + b2*g*g + c1/(g-1.)
    return res
  (dNdgMax, A, thb, fracInMax, energyFracInMax, gMinInd, fail
    ) = subtractMaxwell(gEdges, dNdg, thb)
  dNdgSub = abs(dNdg - dNdgMax)
  gAll = 0.5*(gEdges[1:] + gEdges[:-1])
  dNdgSub = dNdgSub[gMinInd:]
  g = gAll[gMinInd:]
  # get rid of zeros
  nz = (dNdgSub > 0.)
  dNdgSub = dNdgSub[nz]
  g = g[nz]
  # rescale so points are weighted sort of the same
  lndNdgSub = numpy.log(dNdgSub)
  lnScaleFactor = 1. - lndNdgSub.min()
  lndNdgSub += lnScaleFactor
  lndNdgSubMaxInd = lndNdgSub.argmax()
  if 1: #{
    if 1: #ySigmas is None:
      lnySigmas = None
    else:
      lnySigmas = numpy.log(ys+ySigmas) - numpy.log(ys)
    a0 = -1.5
    b10 = -1./g.max()
    b20 = -b10**2
    c10 = -1.
    gc = max(2., g[lndNdgSubMaxInd])
    print lnPowLawExp2(gc, 0., a0, b10, b20, c10)
    lnC0 = lndNdgSub[lndNdgSubMaxInd] - lnPowLawExp2(
                   gc, 0., a0, b10, b20, c10)
    print "gc = %g, lnC0 = %g, lndNdgMax = %g" % (
      gc, lnC0, lndNdgSub[lndNdgSubMaxInd])
    
    #popt = (lnC0, a0, b10, b20, c10)
    fit = scipy.optimize.curve_fit(lnPowLawExp2, 
      g, lndNdgSub, p0=(lnC0,a0,b10,b20,c10), sigma=lnySigmas)
    print fit
    popt, pcov = fit
    (lnC,a,b1,b2,c1) = popt
    lnC -= lnScaleFactor
    C = numpy.exp(lnC)
    # Not sure if this is the right thing to do with lnScaleFactor != 0
    # :TODO: double check
    #pcov[0,:] *= C
    #pcov[:,0] *= C
  #}
  #print popt
  def lnFitFn(g):
    return lnPowLawExp2(g, lnC, a, b1, b2, c1)
  def fitFn(g):
    return numpy.exp(lnFitFn(g))
  if 1: #{
    import pylab
    pylab.loglog(g, dNdgSub, 'o', mfc='none')
    pylab.loglog(g, fitFn(g))
    #pylab.semilogx(g, lnFitFn(g))
    pylab.title("%.3g, %.3g, %.3g, %.3g, %.3g" % (lnC, a, b1, b2, c1))
    pylab.show()
    raise RuntimeError, "Quitting after this diagnostic plot"
  #}
  return (C,a,b1,b2,c1,pcov)
#}

def fit2(gEdges, dNdg, thb): #{
  """
  fits 
  dNdg = C g^a e^{b1 g + b2 g^2 + c1/(g-1)} + MC g sqrt{g^2-1} exp(-g/th) 
  returns C, a, b1, b2, c1, MC, th cov
   where cov is the normalized covariance of the parameters
   let (C, a, b) = (a_0, a_1, a_2)
   Then cov_ij = < [a_i - <a_i>] [a_j - <a_j>] >
   where <a_i> is the value returned (e.g., C, a, or b).
   E.g., sqrt(cov_ii) is the std. dev. of a_i.

   If the ySigmas are known, then the "real" cov can be find by
   real cov = N/(N-M) r^2 cov
   where r^2 = Sum_n (y_n - fitted_y_n(C,a,b))^2 / ySigmas_n^2.
  """
  absb = True
  def lnPowLawExp2(g,lnC,a,a2,b1,b2,c1,d):
    if absb:
      b1 = -abs(b1)
      b2 = -abs(b2)
      c1 = -abs(c1)
      d = min(10., abs(d))
    res = lnC + a*numpy.log(g) + a2*numpy.log(g)**2 + b1*g + b2*g*g 
    res += c1/(g**d)
    return res
  def lnMaxwell(g, lnA, th): 
    res = lnA + numpy.log(g) + 0.5*numpy.log((g-1.)*(g+1.)) - (g-1.)/th
    return res
  def lnPowLawExpPlusMaxwell(g,lnC,a,a2,b1,b2,c1,d,lnMC, th):
    #th = 0.0431
    #d=1.
    a2=0.
    #b2=0.
    #b1=0.
    #b2=0.
    res = numpy.exp(lnPowLawExp2(g,lnC,a,a2,b1,b2,c1,d))
    maxwell = numpy.exp(lnMaxwell(g, lnMC, th))
    res += maxwell
    lnres = numpy.log(res)
    return lnres
  # fit low-E maxwell to get initial parametrs for Maxwellian
  (dNdgMax, MC0, th0, fracInMax, energyFracInMax, gMinInd, fail
    ) = subtractMaxwell(gEdges, dNdg, thb)
  # get rid of zeros
  nz = (dNdg > 0.)
  dNdgNz = dNdg[nz]
  gAll = 0.5*(gEdges[1:] + gEdges[:-1])
  g = gAll[nz]
  # rescale so points are weighted sort of the same
  lndNdg = numpy.log(dNdgNz)
  lnScaleFactor = 1. - lndNdg.min()
  lndNdg += lnScaleFactor
  lnMC0 = numpy.log(MC0) + lnScaleFactor
  lndNdgMaxInd = lndNdg.argmax()
  if 1: #{
    if 1: #ySigmas is None:
      lnySigmas = None
    else:
      lnySigmas = numpy.log(ys+ySigmas) - numpy.log(ys)
    a0 = -1.5
    a20 = -0.01
    b10 = -1./g.max()
    b20 = -b10**2
    c10 = -1.
    d0 = 1.
    lnC0 = numpy.log(dNdg[gMinInd]) - lnPowLawExpPlusMaxwell(
                   gEdges[gMinInd], 0., a0, a20, b10, b20, c10, d0, 0., th0) 
    
    popt = (lnC0, a0, a20, b10, b20, c10, d0, lnMC0, th0)
    fit = scipy.optimize.curve_fit(lnPowLawExpPlusMaxwell, 
      g, lndNdg, p0=popt, sigma=lnySigmas, maxfev = 60000)
    fit = scipy.optimize.curve_fit(lnPowLawExpPlusMaxwell, 
      g, lndNdg, p0=fit[0], sigma=lnySigmas, maxfev = 60000)
    print fit
    popt, pcov = fit
    (lnC,a,a2, b1,b2,c1,d, lnMC, th) = popt
    if absb:
      b1 = -abs(b1)
      b2 = -abs(b2)
      c1 = -abs(c1)
      d = min(10., abs(d))
    lnC -= lnScaleFactor
    lnMC -= lnScaleFactor
    C = numpy.exp(lnC)
    # Not sure if this is the right thing to do with lnScaleFactor != 0
    # :TODO: double check
    #pcov[0,:] *= C
    #pcov[:,0] *= C
  #}
  #print popt
  lndNdg -= lnScaleFactor
  def lnFitFn(g):
    return lnPowLawExpPlusMaxwell(g, lnC, a, a2, b1, b2, c1, d, lnMC, th)
  def fitFn(g):
    return numpy.exp(lnFitFn(g))
  if 0: #{
    avgResidSqr = ((lnFitFn(g) - lndNdg)**2).sum() / g.size
    print "avgResidSqr = %g" % avgResidSqr
    #import pylab
    #pylab.plot(g, lnFitFn(g))
    #pylab.plot(g,lndNdg)
    #pylab.show()
  #}
  if 1: #{
    import pylab
    p = 0.
    pylab.loglog(gAll, gAll**p *dNdg, 'o', mfc='none', mec='c', alpha=0.2)
    pylab.loglog(gAll, gAll**p *fitFn(gAll), '-b')
    pylab.loglog(gAll, gAll**p *fitFn(gAll[gMinInd+40])*(gAll/gAll[gMinInd+40])**a, "--r")
    #pylab.semilogx(g, lnFitFn(g))
    pylab.title("%.3g, a=%.3g, %.3g, -1/b1=%.3g,\n$1/\\sqrt{-b_2}$=%.3g, %.3g, d=%.2g, %.3g, %.3g" % (
      lnC, a, a2, -1./b1, (-b2)**(-0.5), (-c1)**(1./d), d, lnMC, th))
    pylab.xlim(xmax = 2*g[-1])
    pylab.ylim(ymin = 0.5*g[-1]**p * dNdgNz.min(), ymax = 2*g[gMinInd]**p*dNdgNz.max())
    #pylab.gca().set_xscale("linear")
    pylab.subplots_adjust(top=0.86)
    pylab.show()
    raise RuntimeError, "Quitting after this diagnostic plot"
  #}
  return (C,a,b1,b2,c1,pcov)
#}
  
def fitJustPowerAndExp(g, dNdg, b1, b2, lnFitLowEnergy, lnFitPowerLaw): #{
  """
  Given g and dNdg data, excludes the low (Maxwellian) and high
  (steep cutoff) energy parts, and fits the remainder to 
  f(g) = C g^a exp(-b g)
  returns (C, a, b, pCov) 
  
  Assumes that dNdg > 0 so we can take its log.
  """
  numPts = len(g)
  # Find where maxwellian finally falls below power law part
  partsDif = lnFitLowEnergy(g) - lnFitPowerLaw(g)
  iX1 = 1 + numpy.where(partsDif > 0.)[0][-1]
  # Now move to higher energies until we enter a concave-down region.
  lngdif = numpy.diff(numpy.log(g))
  lngdif2 = 0.5*(lngdif[1:] + lngdif[:-1])
  d2 = numpy.diff(numpy.diff(numpy.log(dNdg))/lngdif)/lngdif2
  d2 = arrayPlus.smooth1d(d2, 4)
  concaveDown = (d2 <= 0.)
  dnsInARowMin = 5
  for iX2 in range(iX1, numPts - dnsInARowMin):
    print iX2, concaveDown[iX2:iX2+dnsInARowMin]
    if concaveDown[iX2:iX2+dnsInARowMin].all():
      print "Found!"
      break
  if 0: #{
    import pylab
    pylab.loglog(g, dNdg, 'o')
    pylab.show()
  #}
  if iX2 == numPts - dnsInARowMin - 1:
    msg = "Never found %i points of ln(dNdg) vs. ln(g) in a row that" % dnsInARowMin
    msg += " were concave down"
    print "concaveDns = ", concaveDown
    raise ValueError, msg
  # Find the point at which the tail drops the distribution by dipFactor
  #   exp(b1*g+b2*g^2) = 1/dipFactor
  #   b1 g + b2 g^2 + ln(dipFactor) = 0
  dipFactor = 10.
  # If b1 and b2 are both negative (usual), then roots are of opposite
  # sign, and we take positive.
  # If b1 > 0 (b1 is probably extremely small) and b2 < 0, 
  # roots are of opposite sign and again we want the positive root.
  # If b2 > 0 (b2 is probably extremely small) and b1 < 0,
  # roots are booth positive, and we want the smaller.
  gDip = arrayPlus.quadraticSoln([b2], b1, numpy.log(dipFactor),
    sortBy = "nonnegativeOrAbsSmallestFirst")[:,0]
  iX3 = numpy.where(g>gDip)[0][0]
  # fit power law with exp cutoff
  (C, a, b, powLawExpCov) = mathPlus.fitPowerLawExp(g[iX2:iX3],
    dNdg[iX2:iX3])
  if 1: #{
    import pylab
    pylab.loglog(g, dNdg, 'o')
    pylab.loglog(g, C*g**a * numpy.exp(b*g))
    pylab.title(r"a=%g, b=%g" % (a, b))
    pylab.show()
  #}
  return (C, a, b, powLawExpCov)
#}

def fit3(gEdges, dNdg, thb, dMacro = None, startWithPrevFit = None,
  useCachedData = None, fitPowExp = False): #{
  """
  fits 
  dNdg = C g^a e^{b1 g + b2 g^2 - (c1/g)^d)} 
     + \int_{th1}^{th2} MC/(th2-th1) (1/th) g sqrt{g^2-1} exp(-g/th) dth
  returns C, a, b1, b2, c1, MC, th1, th2 cov
   where cov is the normalized covariance of the parameters
   let (C, a, b) = (a_0, a_1, a_2)
   Then cov_ij = < [a_i - <a_i>] [a_j - <a_j>] >
   where <a_i> is the value returned (e.g., C, a, or b).
   E.g., sqrt(cov_ii) is the std. dev. of a_i.

   If the ySigmas are known, then the "real" cov can be find by
   real cov = N/(N-M) r^2 cov
   where r^2 = Sum_n (y_n - fitted_y_n(C,a,b))^2 / ySigmas_n^2.

  if startWithPrevFit is the popt from a previous call to fit3 
   (presumably with a similar dNdg), then starts with those parameters,
   rather than calling a bunch of initial conditions.
  """
  #useCachedData = None
  #print 'Warning: not using cached data'
  absb = False
  dMax = 10.
  def lnPowLawExp2(g,lnC,a,a2,b1,b2,c1,d):
    if absb:
      b1 = -abs(b1)
      b2 = -abs(b2)
      #c1 = -abs(c1)
      d = min(dMax, abs(d))
    res = lnC + a*numpy.log(g) + a2*numpy.log(g)**2 + b1*g + b2*g*g 
    # Don't let c and d get ridiculous
    d = min(d, 100)
    c1 = min(c1, 10**(100./max(1.,d)) )
    #tmp = (c1/g)**d
    #if not numpy.isfinite(tmp).all():
    #  inds = numpy.where(numpy.logical_not(numpy.isfinite(tmp)))[0]
    #  print c1, d
    #  print g[inds]
    res += -abs(c1/g)**d
    #res += c1/g**d
    return res
  def lnMaxwell(g, lnA, th): 
    res = lnA + numpy.log(g) + 0.5*numpy.log((g-1.)*(g+1.)) - (g-1.)/th
    return res
  def lnIntMaxwell(g, lnA, th1, dth = None): 
    if numpy.isnan(g).any():
      raise RuntimeError, "g is nan"
    if numpy.isnan(lnA).any():
      raise RuntimeError, "lnA is nan"
    if numpy.isnan(th1).any():
      raise RuntimeError, "th1 is nan"
    if dth is not None and numpy.isnan(dth).any():
      raise RuntimeError, "dth is nan"
    if dth is None:
      return lnMaxwell(g,lnA,th1)
    if absb:
      th1 = abs(th1)
      dth = abs(dth)
    th2 = th1 + dth
    if not isinstance(g, numpy.ndarray):
      return lnIntMaxwell(numpy.array([g]), lnA, th1, dth)[0]
    if (g<=1).any():
      raise ValueError, "g =" + repr(g)
    res = lnA + numpy.log(g) + 0.5*numpy.log((g-1.)*(g+1.)) 
    # (d/d th) exp1(a/th) = exp(-a/th) / th 
    # We want to calculate:
    # IntMaxwell = A g sqrt(g^2-1) (thAvg/(th2-th1)) 
    #                 int_{th1}^{th2} exp(- (g-1)/th) / th d th
    #    = A g sqrt(g^2-1) (thAvg/(th2-th1)) [exp((g-1)/th)]_th1^th2
    # where
    #   thAvg = (th1+th2)/2
    # But we need to make sure things work correctly in limits.
    #
    # First, if (th2 - th1) << th1:
    # IntMaxwell = A g sqrt(g^2-1) exp(-(g-1)/thAvg)
    #  (which is why we added (thAvg/(th2-th1)) to the normalization)
    # Second, when (g-1)/th is very large, we run into underflow and
    #  precision loss.
    # The integral int_th1^th2 exp(-(g-1)/th) / th dth
    #   is bounded by:
    #  (lower bound) (th2-th1)/th2 exp(-(g-1)/th1)
    #  (upper bound) (th2-th1)/th1 exp(-(g-1)/th2)
    thAvg = 0.5*(th1+th2)
    lnUpperBound = numpy.log(thAvg/th1) - (g-1.)/th2 if th1 > 0. else -460.5+0*g
    gBig = (lnUpperBound < -345.4) # e^(-345.4) = 1e-150
    gSmall = numpy.logical_not(gBig)
    gs = g[gSmall].copy()
    # exp1(0) = infinity, but expA2-expA1 = 1/th2 - 1/th1
    g1 = (gs==1.)
    gs[g1] = 1. + 1e-6*th1
    expA1 = scipy.special.exp1( (gs-1.)/th1 ) if th1 > 0. else 0.
    expA2 = scipy.special.exp1( (gs-1.)/th2 ) if th2 > 0. else 0.
    expA = (expA2 - expA1) * thAvg / dth if (dth > 0.) else 0.
    expA[g1] = numpy.log(th2/th1) * thAvg/dth if (dth>0. and th1>0.) else 0.
    if 0:
      iBad = numpy.where(numpy.logical_not(numpy.isfinite(expA)))[0]
      if len(iBad) > 0:
        print iBad, th1, th2
        print gs[iBad], gs[iBad]-1.
        print expA1[iBad]
        print expA2[iBad]
    expC = numpy.exp( -(gs-1.)/ thAvg)
    dthThresh = th1/10.
    if th1 > 0. and dth < dthThresh:
      mixParam = dth/dthThresh
      expRes = expC * (1. - mixParam) + expA * mixParam
    else:
      expRes = expA
    res[gBig] += lnUpperBound[gBig]
    res[gSmall] += numpy.log(expRes)
    return res
  def lnPowLawExpPlusMaxwell(g,lnC,a,a2,b1,b2,c1,d,lnMC, th1, dth = None):
    #if dth is None:
    #  dth = 0.001*abs(th2)
    #th = 0.0431
    #d=1.
    #a2=0.
    #b2=0.
    #b1=0.
    #b2=0.
    lnplaw = lnPowLawExp2(g,lnC,a,a2,b1,b2,c1,d)
    lnmax = lnIntMaxwell(g, lnMC, th1, dth)
    if isinstance(lnplaw, numpy.ndarray):
      # avoid overflow
      lnplaw[lnplaw > 460.5] = 460.5 # ln(1e200) = 460
      lnmax[lnmax > 460.5] = 460.5 # ln(1e200) = 460
      # and underflow
      lnplaw[lnplaw < -460.5] = -460.5 # ln(1e-200) = -460
      lnmax[lnmax < -460.5] = -460.5
    else:
      lnplaw = max(-460.5, min(460.5, lnplaw))
      lnmax = max(-460.5, min(460.5, lnmax))
    res = numpy.exp(lnplaw)
    maxwell = numpy.exp(lnmax)
    res += maxwell
    #if (res==0).any():
    #  iBad = numpy.where(res==0)[0]
    #  print "lnplaw", lnplaw[iBad]
    #  print "lnmax", lnmax[iBad] 
    #  print g[iBad]
    #  print (lnC,a,a2,b1,b2,c1,d,lnMC, th1, dth)
    #  raise ValueError
    lnres = numpy.log(res)
    return lnres
  # get rid of zeros
  nz = (dNdg > 0.)
  dNdgNz = dNdg[nz]
  gAll = 0.5*(gEdges[1:] + gEdges[:-1])
  g = gAll[nz]
  if dMacro is not None: #{
    # amount dNdg varies due to counting noise
    dNdgSig = dNdgNz * (1. + 1./numpy.sqrt(dMacro[nz]))
  #}
  # rescale so points are weighted sort of the same
  lndNdg = numpy.log(dNdgNz)
  lnScaleFactor = 1. - lndNdg.min()
  lndNdg += lnScaleFactor
  if 0: #{
    import pylab
    pylab.loglog(g, numpy.exp(lnMaxwell(g,lnMC0, th20)))
    pylab.loglog(g, numpy.exp(lnIntMaxwell(g,lnMC0, th20, dth20)))
    pylab.show()
  #}
  numNonZero = nz.sum()
  if numNonZero < 10: #{
    lnC = -100.
    a = 0.
    a2 = 0.
    b1 = 0.
    b2 = 0.
    c1 = 0.
    d = 1.
    if numNonZero == 0:
      lnMC = 0.
    else:
      lnMC = lndNdg.max()
    th1 = thb
    dth = thb/100.
    popt = [lnC, a, a2, b1, b2, c1, d, lnMC, th1, dth]
    pcov = numpy.zeros((len(popt),)*2)
  elif 1: #}{
    if dMacro is None: #ySigmas is None:
      lnySigmas = None
    else:
      #lnySigmas = None
      lnySigmas = numpy.log(dNdgNz+dNdgSig) - numpy.log(dNdgNz)
    if True: # startWithPrevFit is None:  #{
      # fit low-E maxwell to get initial parameters for Maxwellian
      th10 = thb
      (dNdgMax, MC0, th20, fracInMax, energyFracInMax, gMinInd, fail
        ) = subtractMaxwell(gEdges, dNdg, thb)
      if (MC0 <= 0): #{ can easily happen for power with negative index
        lnMC0 = lndNdg.min() - 3.
      else: #}{
        lnMC0 = numpy.log(MC0) + lnScaleFactor
      #}
      dth0 = abs(th20-th10)
      th10 = min(th10, th20)
      
      a0 = -1.5
      a20 = 0.
      b10 = -1./g.max()
      b20 = -b10**2
      #c10 = -1.
      c10 = 1.
      d0 = 2.
      dth0 = 0.01 * th20
      # occasionally have problems (especially with initial distributions)
      # where dNdg has only a few non-zero elements
      igMax = numpy.argmax(dNdg)
      igUse = gMinInd
      if dNdg[gMinInd] < 0.01 * dNdg[igMax]:
        igUse = igMax
      lnC0 = numpy.log(dNdg[igUse]) - lnPowLawExpPlusMaxwell(
                     0.5*(gEdges[igUse]+gEdges[igUse+1]), 
                     0., a0, a20, b10, b20, c10, d0, 0., th10, dth0) 
      
      if 0: #{
        lnC0 = -1.28+lnScaleFactor
        a0 = -1.18
        b10 = -1./912.
        b20 = -1./1.85e3**2
        c10 = -4.14**10.
        d0 = 10.
        lnMC0 = -2.03+lnScaleFactor
        th20 = 0.807
      #}
    else: #}{
      (lnC0, a0, a20, b10, b20, c10, d0, lnMC0, th10, dth0) = startWithPrevFit
    #}
    lnCInd = 0
    aInd = 1
    dInd = 6
    b1Ind = 3
    b2Ind = b1Ind+1
    c1Ind = 5
    thInd = -2
    dthInd = -1
    popt = [lnC0, a0, a20, b10, b20, c10, d0, lnMC0, th10, dth0]
    pScale = [1., 1., 1., 1./g[-1], 1./g[-1]**2, 1., 0.5, 1., 1., thb] 
    pBnds = [None, [-10.,0.], [0., 0.], [-1.,0.], [-1.,0.],
            [0., g[-1]], [1., dMax], None, [0., g[-1]/3.], 
            [min(1e-5, thb/10.), g[-1]/3.]]
    useCache = False #(useCachedData is not None)
    if useCache: #{
      poptCache = list(useCachedData[:len(popt)])
      popt = list(poptCache)
    #}
    # first optimization, keep d constant -- otherwise, algorithm likes
    #   to increase d without bound for some reason
    if useCache: #{
      starts = [["orig"]]
    else: #}{ 
      starts = [
        ["orig",],
        ["follow",],
        ["orig", [dthInd], [dInd], [thInd, th20], [b2Ind, 0.], [c1Ind, 3*th20]],
        #["follow", [dthInd], [dInd], [b2Ind, 0.]],
        ["follow",],
        ["orig", [dthInd], [dInd], [thInd, th20], [b1Ind, 0.], [c1Ind, 3*th20]],
        #["follow", [dthInd], [dInd], [b1Ind, 0.]],
        ["follow",],
        ["orig", [dthInd], [dInd] ],
        #["follow", [dthInd]],
        ["follow",],
        ["best", [dInd], [c1Ind, 3*th20]],
        ["follow",],
        #["orig", [dInd]],
        #["follow",],
        #["best", [dthInd], [dInd, 2.]],
        #["follow",],
        #["best", [dthInd]],
        #["follow",],
        ["best"],
        ]
      if startWithPrevFit is not None:
        starts.append(["new", startWithPrevFit])
        starts.append(["best"])
      starts.append(["follow"])
    #}
    redo = False
    try:
      (fit, residImprovement) = optPlus.curveFitWithBoundedParamsMultipleStarts(
        scipy.optimize.curve_fit,
        lnPowLawExpPlusMaxwell, g, lndNdg, pScale, pBnds, 
        starts = starts, p0 = popt, sigma = lnySigmas)
    except:
      if useCache:
        redo = True
      else:
        raise
    
    popt, pcov = fit
    (lnC,a,a2, b1,b2,c1,d, lnMC, th1, dth) = popt
    def lnFitLowEnergy(g):
      p = list([lnMC, th1, dth])
      return lnIntMaxwell(g, *p)
    def lnFitPowerLaw(g):
      p = list([lnC, a, a2, b1, b2, c1, d])
      return lnPowLawExp2(g, *p)

    if fitPowExp: #{
      (CPowExp, aPowExp, bPowExp, powExpCov) = fitJustPowerAndExp(
        g, dNdgNz, b1, b2, lnFitLowEnergy, lnFitPowerLaw)
      lnCPowExp = numpy.log(CPowExp)
      starts2 = [
        ["orig", [aInd, aPowExp], [b1Ind, bPowExp], [lnCInd, lnCPowExp]],
        ["follow", [aInd, aPowExp], [b1Ind, bPowExp], [lnCInd, lnCPowExp]],
        ]
      popt[lnCInd] = lnCPowExp
      popt[aInd] = aPowExp
      popt[b1Ind] = bPowExp
      print "popt new", popt
      (fit, residImprovement) = optPlus.curveFitWithBoundedParamsMultipleStarts(
        scipy.optimize.curve_fit,
        lnPowLawExpPlusMaxwell, g, lndNdg, pScale, pBnds, 
        starts = starts2, p0 = popt, sigma = lnySigmas)
      popt, pcov = fit
      (lnC,a,a2, b1,b2,c1,d, lnMC, th1, dth) = popt
      print 'a=%g', a, aPowExp
      def lnFitLowEnergy(g):
        p = list([lnMC, th1, dth])
        return lnIntMaxwell(g, *p)
      def lnFitPowerLaw(g):
        p = list([lnC, a, a2, b1, b2, c1, d])
        return lnPowLawExp2(g, *p)
    #}

    if useCache: #{
      # check that popt hasn't change significantly
      # if it has, run whole optimization
      dif = numpy.array(popt) - numpy.array(poptCache)
      mag = numpy.array([max(abs(a),abs(b)) for (a,b) in zip(popt, poptCache)])
      mag[mag==0.] = 1e-16
      reldif = abs(dif)/mag
      if redo or (dif > 9e-3).any() or (residImprovement > 1.+1e-6):
        res = fit3(gEdges, dNdg, thb, dMacro = dMacro, 
          startWithPrevFit = startWithPrevFit,
          useCachedData = None)
        return res
    #}

    #print fit
    if absb:
      b1 = -abs(b1)
      b2 = -abs(b2)
      #c1 = -abs(c1)
      d = min(dMax, abs(d))
      th1 = abs(th1)
      dth = abs(dth)
    th2 = th1 + dth
    lnC -= lnScaleFactor
    lnMC -= lnScaleFactor
    C = numpy.exp(lnC)
    # Not sure if this is the right thing to do with lnScaleFactor != 0
    # :TODO: double check
    #pcov[0,:] *= C
    #pcov[:,0] *= C
  #}
  lndNdg -= lnScaleFactor
  def lnFitLowEnergy(g):
    p = list([lnMC, th1, dth])
    return lnIntMaxwell(g, *p)
  def lnFitPowerLaw(g):
    p = list([lnC, a, a2, b1, b2, c1, d])
    return lnPowLawExp2(g, *p)
  def lnFitFn(g):
    p = list([lnC, a, a2, b1, b2, c1, d, lnMC, th1, dth])
    return lnPowLawExpPlusMaxwell(g, *p)
  def fitLowEnergy(g):
    return numpy.exp(lnFitLowEnergy(g))
  def fitPowerLaw(g):
    return numpy.exp(lnFitPowerLaw(g))
  def fitFn(g):
    return fitLowEnergy(g) + fitPowerLaw(g)

    #return fit3(gEdges[iX2:iX3+1], dNdg[iX2:iX3], thb, dMacro = dMacro,
    #    startWithPrevFit = startWithPrevFit,
    #    useCachedData = None, deweightBelowG = g[iX2])
  if 0: #{
    avgResidSqr = ((lnFitFn(g) - lndNdg)**2).sum() / g.size
    print "avgResidSqr = %g" % avgResidSqr
    #import pylab
    #pylab.plot(g, lnFitFn(g))
    #pylab.plot(g,lndNdg)
    #pylab.show()
  #}
  if 0: #{
    import pylab
    print "popt =", popt
    p = 0.
    pylab.loglog(gAll, gAll**p *dNdg, 'o', mfc='none', mec='c', alpha=0.3)
    pylab.loglog(gAll, gAll**p *fitFn(gAll), '-b')
    pylab.loglog(gAll, gAll**p *fitFn(gAll[gMinInd+40])* (gAll/ gAll[gMinInd+40])**a, "--r")
    ylimits = pylab.ylim()
    if 1: #{# make parts
      mwell = fitLowEnergy(g)
      plaw = fitPowerLaw(g)
      pylab.loglog(g, mwell, '-', color='0.1')
      pylab.loglog(g, plaw, '--', color='0.1')
    #}
    pylab.ylim(ylimits)

    #pylab.semilogx(g, lnFitFn(g))
    title = "%.3g, a=%.3g, %.3g, -1/b1=%.3g,\n$1/\\sqrt{-b_2}$=%.3g, %.3g, d=%.2g, %.3g, " % (
      lnC, a, a2, -1./b1 if b1!=0. else 1e99, (-b2)**(-0.5) if b2 != 0. else 1e99, 
      c1, d, lnMC)
    title += "th=%.3g-%.3g" % (th1, th2) 
    pylab.xlabel(r"$\gamma$")
    if p==0:
      ylabel = r"$dN/d\gamma$"
    elif p==1:
      ylabel = r"$\gamma dN/d\gamma$"
    else:
      ylabel = r"$\gamma^{%.3g} dN/d\gamma$" % p
    pylab.ylabel(ylabel)
    pylab.title(title)
    pylab.xlim(xmax = 2*g[-1])
    pylab.ylim(ymin = 0.5*g[-1]**p * dNdgNz.min(), ymax = 2*g[gMinInd]**p*dNdgNz.max())
    #pylab.gca().set_xscale("linear")
    pylab.subplots_adjust(top=0.86)
    pylab.show()
    raise RuntimeError, "Quitting after this diagnostic plot"
  #}
  return (popt, pcov, fitFn, fitLowEnergy, fitPowerLaw)
#}

def fit4(gEdges, dNdg, thb, dMacro = None, startWithPrevFit = None,
  useCachedData = None, deweightBelowG = None): #{
  """
  fits 
  dNdg = C g^a e^{b1 g + b2 g^2 - (c1/g)^d)} 
     + \int_{th1}^{th2} MC/(th2-th1) (1/th) g sqrt{g^2-1} exp(-g/th) dth
  returns C, a, b1, b2, c1, MC, th1, th2 cov
   where cov is the normalized covariance of the parameters
   let (C, a, b) = (a_0, a_1, a_2)
   Then cov_ij = < [a_i - <a_i>] [a_j - <a_j>] >
   where <a_i> is the value returned (e.g., C, a, or b).
   E.g., sqrt(cov_ii) is the std. dev. of a_i.

   If the ySigmas are known, then the "real" cov can be find by
   real cov = N/(N-M) r^2 cov
   where r^2 = Sum_n (y_n - fitted_y_n(C,a,b))^2 / ySigmas_n^2.

  if startWithPrevFit is the popt from a previous call to fit3 
   (presumably with a similar dNdg), then starts with those parameters,
   rather than calling a bunch of initial conditions.

  This calls fit3 first, then extracts the power-law and exp decay part,
  fits a simple power-law-exponent to that.  Then it tries to refit
  the whole data with that power-law and exponent set.
  """
  # Get first fit
  (poptInit, pcovInit, fitFnInit, fitLowEnergyInit, fitPowerLawInit) = fit3(
    gEdges, dNdg, thb, dMacro=dMacro, startWithPrevFit=startWithPrevFit, 
    useCachedData=useCachedData)
  gsAll = gEdges

  # find intersection of maxwell and pow law
  partsDif = lnFitLowEnergy(g) - lnFitPowerLaw(g)
  iX = numpy.where(partsDif > 0.)[0][-1]
  # and now find first non-concave up point after that
  useFit = False
  if useFit:
    pts = lnFitFn(g)
  else: 
    pts = dNdgNz
  lngdif = numpy.diff(numpy.log(g))
  lngdif2 = 0.5*(lngdif[1:] + lngdif[:-1])
  d2 = numpy.diff(numpy.diff(pts)/lngdif)/lngdif2
  iX2 = 1 + iX + numpy.where(d2[iX-1:]<=0)[0][0]
  # and find where tails cause factor of 10 dip
  # exp(b1*g+b2*g^2) = 1/dipFactor
  # b1 g + b2 g^2 + ln(dipFactor) = 0
  dipFactor = 10.
  # If b1 and b2 are both negative (usual), then roots are of opposite
  # sign, and we take positive.
  # If b1 > 0 (b1 is probably extremely small) and b2 < 0, 
  # roots are of opposite sign and again we want the positive root.
  # If b2 > 0 (b2 is probably extremely small) and b1 < 0,
  # roots are booth positive, and we want the smaller.
  gDip = arrayPlus.quadraticSoln(numpy.array([b2]), b1, numpy.log(dipFactor),
    sortBy = "nonnegativeOrAbsSmallestFirst")[:,0]
  iX3 = numpy.where(g>gDip)[0][0]
  # fit power law with exp cutoff
  (C, a, b1, powLawExpCov) = mathPlus.fitPowerLawExp(g[iX2:iX3],
    dNdgNz[iX2:iX3])
  print C, a, b1
  #
  useCachedData = None
  print 'Warning: not using cached data'
  absb = False
  dMax = 10.
  def lnPowLawExp2(g,lnC,a,a2,b1,b2,c1,d):
    if absb:
      b1 = -abs(b1)
      b2 = -abs(b2)
      #c1 = -abs(c1)
      d = min(dMax, abs(d))
    res = lnC + a*numpy.log(g) + a2*numpy.log(g)**2 + b1*g + b2*g*g 
    # Don't let c and d get ridiculous
    d = min(d, 100)
    c1 = min(c1, 10**(100./max(1.,d)) )
    #tmp = (c1/g)**d
    #if not numpy.isfinite(tmp).all():
    #  inds = numpy.where(numpy.logical_not(numpy.isfinite(tmp)))[0]
    #  print c1, d
    #  print g[inds]
    res += -abs(c1/g)**d
    #res += c1/g**d
    return res
  def lnMaxwell(g, lnA, th): 
    res = lnA + numpy.log(g) + 0.5*numpy.log((g-1.)*(g+1.)) - (g-1.)/th
    return res
  def lnIntMaxwell(g, lnA, th1, dth = None): 
    if dth is None:
      return lnMaxwell(g,lnA,th1)
    if absb:
      th1 = abs(th1)
      dth = abs(dth)
    th2 = th1 + dth
    if not isinstance(g, numpy.ndarray):
      return lnIntMaxwell(numpy.array([g]), lnA, th1, dth)[0]
    if (g<=1).any():
      raise ValueError, "g =" + repr(g)
    res = lnA + numpy.log(g) + 0.5*numpy.log((g-1.)*(g+1.)) 
    # (d/d th) exp1(a/th) = exp(-a/th) / th 
    # We want to calculate:
    # IntMaxwell = A g sqrt(g^2-1) (thAvg/(th2-th1)) 
    #                 int_{th1}^{th2} exp(- (g-1)/th) / th d th
    #    = A g sqrt(g^2-1) (thAvg/(th2-th1)) [exp((g-1)/th)]_th1^th2
    # where
    #   thAvg = (th1+th2)/2
    # But we need to make sure things work correctly in limits.
    #
    # First, if (th2 - th1) << th1:
    # IntMaxwell = A g sqrt(g^2-1) exp(-(g-1)/thAvg)
    #  (which is why we added (thAvg/(th2-th1)) to the normalization)
    # Second, when (g-1)/th is very large, we run into underflow and
    #  precision loss.
    # The integral int_th1^th2 exp(-(g-1)/th) / th dth
    #   is bounded by:
    #  (lower bound) (th2-th1)/th2 exp(-(g-1)/th1)
    #  (upper bound) (th2-th1)/th1 exp(-(g-1)/th2)
    thAvg = 0.5*(th1+th2)
    lnUpperBound = numpy.log(thAvg/th1) - (g-1.)/th2 if th1 > 0. else -460.5+0*g
    gBig = (lnUpperBound < -345.4) # e^(-345.4) = 1e-150
    gSmall = numpy.logical_not(gBig)
    gs = g[gSmall].copy()
    # exp1(0) = infinity, but expA2-expA1 = 1/th2 - 1/th1
    g1 = (gs==1.)
    gs[g1] = 1. + 1e-6*th1
    expA1 = scipy.special.exp1( (gs-1.)/th1 ) if th1 > 0. else 0.
    expA2 = scipy.special.exp1( (gs-1.)/th2 ) if th2 > 0. else 0.
    expA = (expA2 - expA1) * thAvg / dth if (dth > 0.) else 0.
    expA[g1] = numpy.log(th2/th1) * thAvg/dth if (dth>0. and th1>0.) else 0.
    if 0:
      iBad = numpy.where(numpy.logical_not(numpy.isfinite(expA)))[0]
      if len(iBad) > 0:
        print iBad, th1, th2
        print gs[iBad], gs[iBad]-1.
        print expA1[iBad]
        print expA2[iBad]
    expC = numpy.exp( -(gs-1.)/ thAvg)
    dthThresh = th1/10.
    if th1 > 0. and dth < dthThresh:
      mixParam = dth/dthThresh
      expRes = expC * (1. - mixParam) + expA * mixParam
    else:
      expRes = expA
    res[gBig] += lnUpperBound[gBig]
    res[gSmall] += numpy.log(expRes)
    return res
  def lnPowLawExpPlusMaxwell(g,lnC,a,a2,b1,b2,c1,d,lnMC, th1, dth = None):
    #if dth is None:
    #  dth = 0.001*abs(th2)
    #th = 0.0431
    #d=1.
    #a2=0.
    #b2=0.
    #b1=0.
    #b2=0.
    lnplaw = lnPowLawExp2(g,lnC,a,a2,b1,b2,c1,d)
    lnmax = lnIntMaxwell(g, lnMC, th1, dth)
    if isinstance(lnplaw, numpy.ndarray):
      # avoid overflow
      lnplaw[lnplaw > 460.5] = 460.5 # ln(1e200) = 460
      lnmax[lnmax > 460.5] = 460.5 # ln(1e200) = 460
      # and underflow
      lnplaw[lnplaw < -460.5] = -460.5 # ln(1e-200) = -460
      lnmax[lnmax < -460.5] = -460.5
    else:
      lnplaw = max(-460.5, min(460.5, lnplaw))
      lnmax = max(-460.5, min(460.5, lnmax))
    res = numpy.exp(lnplaw)
    maxwell = numpy.exp(lnmax)
    res += maxwell
    #if (res==0).any():
    #  iBad = numpy.where(res==0)[0]
    #  print "lnplaw", lnplaw[iBad]
    #  print "lnmax", lnmax[iBad] 
    #  print g[iBad]
    #  print (lnC,a,a2,b1,b2,c1,d,lnMC, th1, dth)
    #  raise ValueError
    lnres = numpy.log(res)
    return lnres
  # get rid of zeros
  nz = (dNdg > 0.)
  dNdgNz = dNdg[nz]
  gAll = 0.5*(gEdges[1:] + gEdges[:-1])
  g = gAll[nz]
  if dMacro is not None: #{
    # amount dNdg varies due to counting noise
    dNdgSig = dNdgNz * (1. + 1./numpy.sqrt(dMacro[nz]))
  #}
  # rescale so points are weighted sort of the same
  lndNdg = numpy.log(dNdgNz)
  lnScaleFactor = 1. - lndNdg.min()
  lndNdg += lnScaleFactor
  if 0: #{
    import pylab
    pylab.loglog(g, numpy.exp(lnMaxwell(g,lnMC0, th20)))
    pylab.loglog(g, numpy.exp(lnIntMaxwell(g,lnMC0, th20, dth20)))
    pylab.show()
  #}
  numNonZero = nz.sum()
  if numNonZero < 10: #{
    lnC = -100.
    a = 0.
    a2 = 0.
    b1 = 0.
    b2 = 0.
    c1 = 0.
    d = 1.
    if numNonZero == 0:
      lnMC = 0.
    else:
      lnMC = lndNdg.max()
    th1 = thb
    dth = thb/100.
    popt = [lnC, a, a2, b1, b2, c1, d, lnMC, th1, dth]
    pcov = numpy.zeros((len(popt),)*2)
  elif 1: #}{
    if dMacro is None: #ySigmas is None:
      lnySigmas = None
    else:
      #lnySigmas = None
      lnySigmas = numpy.log(dNdgNz+dNdgSig) - numpy.log(dNdgNz)
    if True: # startWithPrevFit is None:  #{
      # fit low-E maxwell to get initial parametrs for Maxwellian
      th10 = thb
      (dNdgMax, MC0, th20, fracInMax, energyFracInMax, gMinInd, fail
        ) = subtractMaxwell(gEdges, dNdg, thb)
      lnMC0 = numpy.log(MC0) + lnScaleFactor
      dth0 = abs(th20-th10)
      th10 = min(th10, th20)
      
      if deweightBelowG is not None: #{ de-weight part below 3*thb
        print 'deweighting above g = %g' % deweightBelowG
        lnySigmas[g<deweightBelowG] *= 100.
      #}

      a0 = -1.5
      a20 = 0.
      b10 = -1./g.max()
      b20 = -b10**2
      #c10 = -1.
      c10 = 1.
      d0 = 2.
      dth0 = 0.01 * th20
      # occasionally have problems (especially with initial distributions)
      # where dNdg has only a few non-zero elements
      igMax = numpy.argmax(dNdg)
      igUse = gMinInd
      if dNdg[gMinInd] < 0.01 * dNdg[igMax]:
        igUse = igMax
      lnC0 = numpy.log(dNdg[igUse]) - lnPowLawExpPlusMaxwell(
                     0.5*(gEdges[igUse]+gEdges[igUse+1]), 
                     0., a0, a20, b10, b20, c10, d0, 0., th10, dth0) 
      
      if 0: #{
        lnC0 = -1.28+lnScaleFactor
        a0 = -1.18
        b10 = -1./912.
        b20 = -1./1.85e3**2
        c10 = -4.14**10.
        d0 = 10.
        lnMC0 = -2.03+lnScaleFactor
        th20 = 0.807
      #}
    else: #}{
      (lnC0, a0, a20, b10, b20, c10, d0, lnMC0, th10, dth0) = startWithPrevFit
    #}
    dInd = 6
    b1Ind = 3
    b2Ind = b1Ind+1
    c1Ind = 5
    thInd = -2
    dthInd = -1
    popt = [lnC0, a0, a20, b10, b20, c10, d0, lnMC0, th10, dth0]
    pScale = [1., 1., 1., 1./g[-1], 1./g[-1]**2, 1., 0.5, 1., 1., thb] 
    pBnds = [None, [-10.,0.], [0., 0.], [-1.,0.], [-1.,0.],
            [0., g[-1]], [1., dMax], None, [0., g[-1]/3.], 
            [min(1e-5, thb/10.), g[-1]/3.]]
    useCache = (useCachedData is not None)
    if useCache: #{
      poptCache = list(useCachedData[:len(popt)])
      popt = list(poptCache)
    #}
    # first optimization, keep d constant -- otherwise, algorithm likes
    #   to increase d without bound for some reason
    if useCache: #{
      starts = [["orig"]]
    else: #}{ 
      starts = [
        ["orig",],
        ["follow",],
        ["orig", [dthInd], [dInd], [thInd, th20], [b2Ind, 0.], [c1Ind, 3*th20]],
        #["follow", [dthInd], [dInd], [b2Ind, 0.]],
        ["follow",],
        ["orig", [dthInd], [dInd], [thInd, th20], [b1Ind, 0.], [c1Ind, 3*th20]],
        #["follow", [dthInd], [dInd], [b1Ind, 0.]],
        ["follow",],
        ["orig", [dthInd], [dInd] ],
        #["follow", [dthInd]],
        ["follow",],
        ["best", [dInd], [c1Ind, 3*th20]],
        ["follow",],
        #["orig", [dInd]],
        #["follow",],
        #["best", [dthInd], [dInd, 2.]],
        #["follow",],
        #["best", [dthInd]],
        #["follow",],
        ["best"],
        ]
      if startWithPrevFit is not None:
        starts.append(["new", startWithPrevFit])
        starts.append(["best"])
      starts.append(["follow"])
    #}
    redo = False
    try:
      (fit, residImprovement) = optPlus.curveFitWithBoundedParamsMultipleStarts(
        scipy.optimize.curve_fit,
        lnPowLawExpPlusMaxwell, g, lndNdg, pScale, pBnds, 
        starts = starts, p0 = popt, sigma = lnySigmas)
    except:
      if useCache:
        redo = True
      else:
        raise
    
    popt, pcov = fit
    if useCache: #{
      # check that popt hasn't change significantly
      # if it has, run whole optimization
      dif = numpy.array(popt) - numpy.array(poptCache)
      mag = numpy.array([max(abs(a),abs(b)) for (a,b) in zip(popt, poptCache)])
      mag[mag==0.] = 1e-16
      reldif = abs(dif)/mag
      if redo or (dif > 9e-3).any() or (residImprovement > 1.+1e-6):
        res = fit3(gEdges, dNdg, thb, dMacro = dMacro, 
          startWithPrevFit = startWithPrevFit,
          useCachedData = None)
        return res
    #}

    #print fit
    (lnC,a,a2, b1,b2,c1,d, lnMC, th1, dth) = popt
    if absb:
      b1 = -abs(b1)
      b2 = -abs(b2)
      #c1 = -abs(c1)
      d = min(dMax, abs(d))
      th1 = abs(th1)
      dth = abs(dth)
    th2 = th1 + dth
    lnC -= lnScaleFactor
    lnMC -= lnScaleFactor
    C = numpy.exp(lnC)
    # Not sure if this is the right thing to do with lnScaleFactor != 0
    # :TODO: double check
    #pcov[0,:] *= C
    #pcov[:,0] *= C
  #}
  lndNdg -= lnScaleFactor
  def lnFitLowEnergy(g):
    p = list([lnMC, th1, dth])
    return lnIntMaxwell(g, *p)
  def lnFitPowerLaw(g):
    p = list([lnC, a, a2, b1, b2, c1, d])
    return lnPowLawExp2(g, *p)
  def lnFitFn(g):
    p = list([lnC, a, a2, b1, b2, c1, d, lnMC, th1, dth])
    return lnPowLawExpPlusMaxwell(g, *p)
  def fitLowEnergy(g):
    return numpy.exp(lnFitLowEnergy(g))
  def fitPowerLaw(g):
    return numpy.exp(lnFitPowerLaw(g))
  def fitFn(g):
    return fitLowEnergy(g) + fitPowerLaw(g)

  if deweightBelowG is None and useCachedData is None:
    # find intersection of maxwell and pow law
    partsDif = lnFitLowEnergy(g) - lnFitPowerLaw(g)
    iX = numpy.where(partsDif > 0.)[0][-1]
    # and now find first non-concave up point after that
    useFit = False
    if useFit:
      pts = lnFitFn(g)
    else: 
      pts = dNdgNz
    lngdif = numpy.diff(numpy.log(g))
    lngdif2 = 0.5*(lngdif[1:] + lngdif[:-1])
    d2 = numpy.diff(numpy.diff(pts)/lngdif)/lngdif2
    iX2 = 1 + iX + numpy.where(d2[iX-1:]<=0)[0][0]
    # and find where tails cause factor of 10 dip
    # exp(b1*g+b2*g^2) = 1/dipFactor
    # b1 g + b2 g^2 + ln(dipFactor) = 0
    dipFactor = 10.
    # If b1 and b2 are both negative (usual), then roots are of opposite
    # sign, and we take positive.
    # If b1 > 0 (b1 is probably extremely small) and b2 < 0, 
    # roots are of opposite sign and again we want the positive root.
    # If b2 > 0 (b2 is probably extremely small) and b1 < 0,
    # roots are booth positive, and we want the smaller.
    gDip = arrayPlus.quadraticSoln(numpy.array([b2]), b1, numpy.log(dipFactor),
      sortBy = "nonnegativeOrAbsSmallestFirst")[:,0]
    iX3 = numpy.where(g>gDip)[0][0]
    # fit power law with exp cutoff
    (C, a, b1, powLawExpCov) = mathPlus.fitPowerLawExp(g[iX2:iX3],
      dNdgNz[iX2:iX3])
    print C, a, b1


    #return fit3(gEdges[iX2:iX3+1], dNdg[iX2:iX3], thb, dMacro = dMacro,
    #    startWithPrevFit = startWithPrevFit,
    #    useCachedData = None, deweightBelowG = g[iX2])
  if 0: #{
    avgResidSqr = ((lnFitFn(g) - lndNdg)**2).sum() / g.size
    print "avgResidSqr = %g" % avgResidSqr
    #import pylab
    #pylab.plot(g, lnFitFn(g))
    #pylab.plot(g,lndNdg)
    #pylab.show()
  #}
  if 0: #{
    import pylab
    print "popt =", popt
    p = 0.
    pylab.loglog(gAll, gAll**p *dNdg, 'o', mfc='none', mec='c', alpha=0.3)
    pylab.loglog(gAll, gAll**p *fitFn(gAll), '-b')
    pylab.loglog(gAll, gAll**p *fitFn(gAll[gMinInd+40])* (gAll/ gAll[gMinInd+40])**a, "--r")
    ylimits = pylab.ylim()
    if 1: #{# make parts
      mwell = fitLowEnergy(g)
      plaw = fitPowerLaw(g)
      pylab.loglog(g, mwell, '-', color='0.1')
      pylab.loglog(g, plaw, '--', color='0.1')
    #}
    pylab.ylim(ylimits)

    #pylab.semilogx(g, lnFitFn(g))
    title = "%.3g, a=%.3g, %.3g, -1/b1=%.3g,\n$1/\\sqrt{-b_2}$=%.3g, %.3g, d=%.2g, %.3g, " % (
      lnC, a, a2, -1./b1 if b1!=0. else 1e99, (-b2)**(-0.5) if b2 != 0. else 1e99, 
      c1, d, lnMC)
    title += "th=%.3g-%.3g" % (th1, th2) 
    pylab.xlabel(r"$\gamma$")
    if p==0:
      ylabel = r"$dN/d\gamma$"
    elif p==1:
      ylabel = r"$\gamma dN/d\gamma$"
    else:
      ylabel = r"$\gamma^{%.3g} dN/d\gamma$" % p
    pylab.ylabel(ylabel)
    pylab.title(title)
    pylab.xlim(xmax = 2*g[-1])
    pylab.ylim(ymin = 0.5*g[-1]**p * dNdgNz.min(), ymax = 2*g[gMinInd]**p*dNdgNz.max())
    #pylab.gca().set_xscale("linear")
    pylab.subplots_adjust(top=0.86)
    pylab.show()
    raise RuntimeError, "Quitting after this diagnostic plot"
  #}
  return (popt, pcov, fitFn, fitLowEnergy, fitPowerLaw)
#}

def gMinMaxForTailFit(gEdges, dN, 
  arbitrariness = {"gMin":1., "gMax":1.}, minPts = None): #{
  """
  dN = dN/dg * dg
  N = dN.sum()
  """
  gAboveAvg = 2. * arbitrariness["gMin"]
  defCutoff = 1e-3
  gMaxCutoff = defCutoff * arbitrariness["gMax"]
  if gMaxCutoff >= 1:
    msg = 'arbitrariness["gMax"] must be less than %.2g' % (1./defCutoff)
    raise ValueError, msg
  g = 0.5*(gEdges[1:] + gEdges[:-1])
  gAvg = (g*dN).sum() / dN.sum()
  igAvgWhere = numpy.where(gEdges <= gAvg)[0]
  igAvg = igAvgWhere[-1]
  gMin = gAboveAvg * gAvg
  igMinWhere = numpy.where(gEdges <= gMin)[0]
  if len(igMinWhere) == 0: # should never happen? (small arrays?)
    igMin = 0
  else:
    igMin = min(igMinWhere[-1], len(gEdges)-2)
  dNdg = dN[igAvg:] / numpy.diff(gEdges[igAvg:])
  igMaxWhere = numpy.where(dNdg < gMaxCutoff * dNdg[0])[0]
  if len(igMaxWhere) == 0:
    igMax = dN.size
  else:
    igMax = igAvg + igMaxWhere[0] 
  if igMax <= igMin:
    igMax = igMin + 1
  gMax = gEdges[igMax]
  
  if minPts is not None and igMax - igMin < minPts: #{
    igMin = max(0, igMin - (minPts - (igMax-igMin))/2)
    igMax = min(len(g), igMin + minPts)
    gMin = gEdges[igMin]
    gMax = gEdges[igMax]
  #}
  # g[igMin] is included, g[igMax] is excluded
  return (gMin, gMax, igMin, igMax)
#}

def estimateDistError(h, dMacro, gammaBinEdges, oneSigmaPct = 0.05): #{
  """
  ad hoc suggest systematic 1 sigma = 5% error,
    plus 1 sigma = 1/sqrt(N) shot/counting noise
  h[ti,bi] is the particles in bin bi at time ti
  dMacro[ti,bi] is the number of macroparticles
  Bin bi goes from gammaBinEdges[bi] to gammaBinEdges[bi+1]

  Returns hStdRel[ti,bi] ( = hSigma[ti,bi] / h[ti,bi,0] )
  """
  z = (dMacro == 0)
  if z.any():
    dMacro = dMacro.copy()
  dMacro[z] = 1.
  # sigma (relative) = 5% + 1/sqrt(N_macro)
  hStdRel = (oneSigmaPct + 1./numpy.sqrt(dMacro)) 
  # avoid zero relative error
  hStdRel[z] = hStdRel[hStdRel > 0].max()
  # smooth relative err since it shouldn't be too noisy
  hStdRel = arrayPlus.smooth1d(hStdRel, 3, axis=1)
  if 0: #{
    import pylab
    g = 0.5*(gammaBinEdges[1:] + gammaBinEdges[:-1])
    dg = numpy.diff(gammaBinEdges)[numpy.newaxis,:]
    dNdg = h[:,:,0] / dg
    dNdgErr = hStdRel * dNdg
    tii = 0
    print "tii = %i" % tii
    plotCmd = pylab.loglog
    def pc(x, y, *args): #{
      p = 1.5
      plotCmd(x, y * x**p, *args)
    #}
    pc(g, dNdg[tii,:], '.r')
    nstd = 3
    pc(gC, dNdgC[tii,:] - nstd*dNdgErr[tii,:], '--b')
    pc(gC, dNdgC[tii,:] + nstd*dNdgErr[tii,:], '--b')
    pylab.show()
    raise RuntimeError, "Quitting after diagnostic plot"
  #}
  return hStdRel
#}


def fit5(gEdges, dNdg, dMacro, stdRel,
  arbitrariness = {"gMin":1., "gMax":1.}): #{
  """
  fit power-law and cutoff to g > gMin only
  """
  if stdRel is None:
    stdRel = estimateDistError(dNdg, dMacro, gEdges)

  # First, truncate spectrum
  #arbitrariness["gMax"] = 0.1
  (gMin, gMax, igMin, igMax) = gMinMaxForTailFit(gEdges, 
    dNdg*numpy.diff(gEdges), arbitrariness = arbitrariness)
  print "Using gMin = %.3g, gMax = %.3g for fitting" % (gMin, gMax), igMin, igMax

  NT = igMax - igMin
  gEdgesT = gEdges[igMin:]
  dNdgT = dNdg[igMin:]
  gEdgesT = gEdges[igMin:igMax+1]
  dNdgT = dNdg[igMin:igMax]
  g = 0.5*(gEdges[1:] + gEdges[:-1])
  gT = g[igMin:igMax]
  lndNdgT = numpy.log(dNdgT)


  if 0: #{
    import pylab
    pylab.loglog(gT, dNdgT)
    pylab.show()
    raise RuntimeError, "stopped after debugging plot"
  #}

  # Now fit.
  # Fit power law + exp
  def lnPowLawExp(x, lnC, a, b): 
    # fit C x^a exp(-b x)
    return lnC + a * numpy.log(x) - b*x 
  # Fit power law + double-exp
  def lnPowLawDblExp(x, lnC, a, b): 
    # fit C x^a exp(-b x**2)
    return lnC + a * numpy.log(x) - b*x**2
  # Fit power law + var-exp
  def lnPowLawVarExp(x, lnC, a, b, c): 
    # fit C x^a exp(-b x**c)
    return lnC + a * numpy.log(x) - b*x**c
  def lnPowLawExpDblExp(x, lnC, a, b1, b2): 
    # fit C x^a exp(-b1 x - b2 x^2)
    return lnC + a * numpy.log(x) - x*(b1 + b2*x)

  def chiSqr(fn, xs, ys, p0, sigmas = None): #{
    N = len(xs)
    if sigmas is None:
      sigmas = 1.
    yFits = fn(xs, *p0)
    res = ( ((ys - yFits)/ sigmas)**2 ).sum()
    nu = N - len(p0)
    # probability, given sigmas, that chi^2 > the measured value
    Q = scipy.special.gammaincc(0.5 * nu , 0.5 * res)
    # find average to allow some comparison
    res /= nu
    return (res, Q)
  #}

  def fitToLower(lnfn, gs, lnfs, igStart, relVar, p0, pcov = None, sigmas = None,
    ): #{
    chi0 = numpy.sqrt(chiSqr(lnfn, gs[igStart:], lnfs[igStart:], p0, sigmas)[0])
    print "chi0^2 = %.2g, p0 = %s" % (chi0**2, str(p0))
    def getCmp(ig): #{
      return relVar * numpy.log(gs[-1]/gs[igStart])
    #}
    def getChiCmp(cmpFactor): #{
      #return cmpFactor
      return numpy.sqrt(cmpFactor**2 + chi0**2)
    #}
    popts = [p0]
    pcovs = [pcov]
    chis = [chi0]
    belowErr = [True]
    for ig in range(igStart-1, -1, -1): #{
      if len(p0) < 4: #{
        (popt, pcov) = scipy.optimize.curve_fit(lnfn, gs[ig:], lnfs[ig:],
          p0 = popts[0], sigma = None, maxfev=defMaxFev)
      else: #}{
        (popt, pcov) = optPlus.curveFitWithBoundedParams(scipy.optimize.curve_fit, 
          lnfn, gs[ig:], lnfs[ig:], 
          paramMags = [1., 1., 1e-4, 1e-8],
          paramBounds = [[-1e200,1e200], [-50.,3.], [0.,1.],[0,1.]],
          p0 = popts[0], sigma = None, maxfev=defMaxFev)
        # I don't know why pcov is sometimes inf -- but let's not accept
        #  such fits
        #if not numpy.isfinite(pcov).any():
        #  msg = "ig = %i, infinite pcov" % ig
        #  print msg
        #  #raise RuntimeError, msg
      #}
      chi = numpy.sqrt(chiSqr(lnfn, gs[ig:], lnfs[ig:], popt, sigmas)[0])
      cmpFactor = getCmp(ig)
      chiCmpFactor = getChiCmp(cmpFactor)
      #print ig, gs[ig], lnfn(gs[ig], *popt), lnfs[ig]
      #print "  popt =", popt
      #print chiCmpFactor, chi,  cmpFactor, abs(lnfn(gs[ig], *popt) - lnfs[ig])
      dev = abs(lnfn(gs[ig:igStart], *popt) - lnfs[ig:igStart]) 
      popts.insert(0, popt)
      pcovs.insert(0, pcov)
      chis.insert(0, chi)
      belowErr.insert(0, 
        ( 
        chi <= chiCmpFactor and 
        dev.max() <= cmpFactor and
          numpy.isfinite(pcovs[0]).all()))
    #}
    print "belowErr:", numpy.int32(belowErr)
    be = numpy.where(belowErr)[0]
    igMin = be[0]
    res = (igMin, gs[igMin], popts[igMin], pcovs[igMin], chis[igMin]**2)
    return res
  #}

  # First, estimate sigmas -- independent of arbitrariness, so we ca
  #  compare
  if 0: #{
    (gMinS, gMaxS, igMinS, igMaxS) = gMinMaxForTailFit(gEdges, 
      dNdg*numpy.diff(gEdges), arbitrariness = {"gMin":1.5, "gMax":10.})
    gS = g[igMinS:igMaxS]
    dNdgS = dNdg[igMinS:igMaxS]
    lndNdgS = numpy.log(dNdgS)
    NS = gS.size
    # power law * exp
    (C0, a0, b0, pcovb) = mathPlus.fitPowerLawExp(gS, dNdgS, fit = "log")
    lnC0 = numpy.log(C0)
    p1 = (lnC0, a0, abs(b0))
    (popt1, pcov1) = scipy.optimize.curve_fit(lnPowLawExp, gS, lndNdgS,
      p0 = p1, maxfev=defMaxFev)
    sigEst1 = numpy.sqrt(chiSqr(lnPowLawExp, gS, lndNdgS, popt1)[0])
    p2 = (lnC0, a0, numpy.sqrt(abs(b0)))
    (popt2, pcov2) = scipy.optimize.curve_fit(lnPowLawDblExp, gS, lndNdgS,
      p0 = p2, maxfev=defMaxFev)
    sigEst2 = numpy.sqrt(chiSqr(lnPowLawDblExp, gS, lndNdgS, popt2)[0])
    p3a = tuple(popt1) + (1.,)
    (popt3a, pcov3a) = scipy.optimize.curve_fit(lnPowLawVarExp, gS, lndNdgS,
      p0 = p3a, maxfev=defMaxFev)
    sigEst3a = numpy.sqrt(chiSqr(lnPowLawVarExp, gS, lndNdgS, popt3a)[0])
    p3b = tuple(popt2) + (2.,)
    (popt3b, pcov3b) = scipy.optimize.curve_fit(lnPowLawVarExp, gS, lndNdgS,
      p0 = p3b, maxfev=defMaxFev)
    sigEst3b = numpy.sqrt(chiSqr(lnPowLawVarExp, gS, lndNdgS, popt3b)[0])
    sigEst3 = min(sigEst3a, sigEst3b)
    print "sigEsts", str([sigEst1, sigEst2, sigEst3])
    sigEst = min([sigEst1, sigEst2, sigEst3])
    dNdgStdRel = numpy.exp(sigEst) - 1.
    print "Estimated sigma (rel.) = %.3g" % dNdgStdRel

    # now estimate again
    stdRel = estimateDistError(dNdg, dMacro, gEdges, oneSigmaPct = dNdgStdRel)
  
    lndNdgStd = numpy.log(1. + stdRel)
    lndNdgStdT = lndNdgStd[igMin:igMax]
    
    print chiSqr(lnPowLawVarExp, gS, lndNdgS, popt3a, 
      sigmas = lndNdgStd[igMinS:igMaxS], ddof=3)
    print chiSqr(lnPowLawVarExp, gT, lndNdgT, popt3a, 
      sigmas = lndNdgStdT, ddof=3)

  #}
  lndNdgStd = None
  lndNdgStdT = None

  # power law * exp
  (C0, a0, b0, pcovb) = mathPlus.fitPowerLawExp(gT, dNdgT, 
    ySigmas = None, fit = "log")
  lnC0 = numpy.log(C0)
  p1 = (lnC0, a0, abs(b0))
  (popt1, pcov1) = scipy.optimize.curve_fit(lnPowLawExp, gT, lndNdgT,
    p0 = p1, sigma = None, maxfev=defMaxFev)
  p2 = (lnC0, a0, numpy.sqrt(abs(b0)))
  (popt2, pcov2) = scipy.optimize.curve_fit(lnPowLawDblExp, gT, lndNdgT,
    p0 = p2, sigma = None, maxfev=defMaxFev)
  
  def curveFit4(gs, lnfs, p0=None, sigma=None, alpha=None): #{
    if p0 is None: #{
      p4a = tuple(popt1) + (0.,)
      p4b = tuple(popt2[:-1]) + (0.,) + tuple(popt2[-1:])
      chi4a = chiSqr(lnPowLawExpDblExp, gs, lnfs, p4a, sigmas = sigma)[0]
      chi4b = chiSqr(lnPowLawExpDblExp, gs, lnfs, p4b, sigmas = sigma)[0]
      #print "init chi4a,b = %.3g, %.3g" % (chi4a, chi4b)
      resa = curveFit4(gs, lnfs, p0=p4a, sigma=sigma, alpha=alpha)
      resb = curveFit4(gs, lnfs, p0=p4b, sigma=sigma, alpha=alpha)
      chi4a = chiSqr(lnPowLawExpDblExp, gs, lnfs, resa[0], sigmas = sigma)[0]
      chi4b = chiSqr(lnPowLawExpDblExp, gs, lnfs, resb[0], sigmas = sigma)[0]
      #print "     chi4a,b = %.3g, %.3g" % (chi4a, chi4b)
      res = resa if chi4a <= chi4b else resb
    else: #}{
      naRange = [-50,3.] if alpha is None else [-alpha, -alpha]
      print "naRange =", naRange
      res = optPlus.curveFitWithBoundedParams(scipy.optimize.curve_fit, 
        lnPowLawExpDblExp, gs, lnfs, 
        paramMags = [1., 1., 1e-3, 1e-6],
        paramBounds = [[-1e200,1e200], naRange, [0.,1.],[0,1.]],
        p0=p0, maxfev=defMaxFev)
      #print "curveFit4: popt =", res[0]
    #}
    return res

  #}
  if 1: #{
    popt4, pcov4 = curveFit4(gT, lndNdgT)
    chi4,q4 = chiSqr(lnPowLawExpDblExp, gT, lndNdgT, popt4, sigmas = lndNdgStdT)
  else: #}{
    p4a = tuple(popt1) + (0.,)
    (popt4a, pcov4a) = scipy.optimize.curve_fit(lnPowLawExpDblExp, gT, lndNdgT,
      p0 = p4a, sigma = None, maxfev=defMaxFev)
    p4b = tuple(popt2[:-1]) + (0.,) + tuple(popt2[-1:])
    (popt4b, pcov4b) = scipy.optimize.curve_fit(lnPowLawExpDblExp, gT, lndNdgT,
    p0 = p4b, sigma = None, maxfev=defMaxFev)
    chi4a,q4a = chiSqr(lnPowLawExpDblExp, gT, lndNdgT, popt4a, sigmas = lndNdgStdT)
    chi4b,q4b = chiSqr(lnPowLawExpDblExp, gT, lndNdgT, popt4b, sigmas = lndNdgStdT)
    if chi4a <= chi4b:
      chi4 = chi4a
      q4 = q4a
      popt4 = popt4a
      pcov4 = pcov4a
    else:
      chi4 = chi4b
      q4 = q4b
      popt4 = popt4b
      pcov4 = pcov4b
  #}
  
  p3a = tuple(popt1) + (1.,)
  (popt3a, pcov3a) = scipy.optimize.curve_fit(lnPowLawVarExp, gT, lndNdgT,
    p0 = p3a, sigma = None, maxfev=defMaxFev)
  p3b = tuple(popt2) + (2.,)
  (popt3b, pcov3b) = scipy.optimize.curve_fit(lnPowLawVarExp, gT, lndNdgT,
    p0 = p3b, sigma = None, maxfev=defMaxFev)
  chi3a,q3a = chiSqr(lnPowLawVarExp, gT, lndNdgT, popt3a, sigmas = lndNdgStdT)
  chi3b,q3b = chiSqr(lnPowLawVarExp, gT, lndNdgT, popt3b, sigmas = lndNdgStdT)

  if chi3a <= chi3b:
    chi3 = chi3a
    q3 = q3a
    popt3 = popt3a
    pcov3 = pcov3a
  else:
    chi3 = chi3b
    q3 = q3b
    popt3 = popt3b
    pcov3 = pcov3b

  
  fitToLowerGamma = True
  if fitToLowerGamma: #{ fit to lower gamma
    relVar = 0.02
    lndNdgL = numpy.log(dNdg[:igMax])
    (igMin1, gMin1, popt1L, pcov1L, chi1L) = fitToLower(
      lnPowLawExp, g[:igMax], lndNdgL, igMin, relVar, popt1, pcov1)
    (igMin2, gMin2, popt2L, pcov2L, chi2L) = fitToLower(
      lnPowLawDblExp, g[:igMax], lndNdgL, igMin, relVar, popt2, pcov2)
    print "popt4 =", popt4, pcov4
    (igMin4, gMin4, popt4L, pcov4L, chi4L) = fitToLower(
      lnPowLawExpDblExp, g[:igMax], lndNdgL, igMin, relVar, popt4, pcov4)
    print "popt4L =", popt4L, pcov4L
    if 0: #{
      p4a = tuple(popt1) + (0.,)
      (igMin4a, gMin4a, popt4La, pcov4La, chi4La) = fitToLower(
        lnPowLawExpDblExp, g[:igMax], lndNdgL, igMin, relVar, p4a, None)
      p4b = tuple(popt2[:-1]) + (0.,) + tuple(popt2[-1:])
      (igMin4b, gMin4b, popt4Lb, pcov4Lb, chi4Lb) = fitToLower(
        lnPowLawExpDblExp, g[:igMax], lndNdgL, igMin, relVar, p4b, None)
      print "4: ", [igMin4, igMin4a, igMin4b]
      i4m = numpy.argmin([igMin4, igMin4a, igMin4b])
      igMin4 = [igMin4, igMin4a, igMin4b][i4m]
      gMin4 = [gMin4, gMin4a, gMin4b][i4m]
      popt4L = [popt4L, popt4La, popt4Lb][i4m]
      pcov4L = [pcov4L, pcov4La, pcov4Lb][i4m]
      chi4L = [chi4L, chi4La, chi4Lb][i4m]
    #}
    alpha1L = -popt1L[1]
    alpha2L = -popt2L[1]
    alpha4L = -popt4L[1]
    if 1: #{
      print "Starting at g=%.2g, for relVar = %.2g%%, measured" % (
        g[igMin], relVar*100)
      print "  1: alpha = %.3g down to g = %.2g" % (alpha1L, gMin1)
      print "  2: alpha = %.3g down to g = %.2g" % (alpha2L, gMin2)
      print "  4: alpha = %.3g down to g = %.2g" % (alpha4L, gMin4)
    #}
    if 1: #{
      print "...Replacing with fits to lower g"
      popt1 = popt1L
      popt2 = popt2L
      popt4 = popt4L
      pcov1 = pcov1L
      pcov2 = pcov2L
      pcov4 = pcov4L
    #}
  #}
  
  
  chi1,q1 = chiSqr(lnPowLawExp, gT, lndNdgT, popt1, sigmas = lndNdgStdT)
  chi2,q2 = chiSqr(lnPowLawDblExp, gT, lndNdgT, popt2, sigmas = lndNdgStdT)
  chi3,q3 = chiSqr(lnPowLawVarExp, gT, lndNdgT, popt3, sigmas = lndNdgStdT)
  chi4,q4 = chiSqr(lnPowLawExpDblExp, gT, lndNdgT, popt4, sigmas = lndNdgStdT)
  
  maxShotErr = 0.20
  igMax2Where = numpy.where(dMacro[igMin:] < 1./maxShotErr**2)[0]
  if len(igMax2Where) == 0:
    igMax2 = dNdg.size
  else:
    igMax2 = igMin + igMax2Where[0]
  lndNdgm = numpy.log(dNdg[igMin:igMax2])
  #lndNdgStdm = lndNdgStd[igMin:igMax2]
  lndNdgStdm = None
  gm = g[igMin:igMax2]
  MT = igMax2 - igMin
  
  chiTail1,qt1 = chiSqr(lnPowLawExp, gm, lndNdgm, popt1, sigmas = lndNdgStdm)
  chiTail2,qt2 = chiSqr(lnPowLawDblExp, gm, lndNdgm, popt2, sigmas = lndNdgStdm)
  chiTail3,qt3 = chiSqr(lnPowLawVarExp, gm, lndNdgm, popt3, sigmas = lndNdgStdm)
  chiTail4,qt4 = chiSqr(lnPowLawExpDblExp, gm, lndNdgm, popt4, sigmas = lndNdgStdm)

  def f1(x):
    res = lnPowLawExp(x, *popt1)
    return res
  def f2(x):
    res = lnPowLawDblExp(x, *popt2)
    return res
  def f3(x):
    res = lnPowLawVarExp(x, *popt3)
    return res
  def f4(x):
    res = lnPowLawExpDblExp(x, *popt4)
    return res
  def f5(x):
    res = lnPowLawExpDblExp(x, *popt5)
    return res

  alpha1 = -popt1[1]
  alpha2 = -popt2[1]
  alpha3 = -popt3[1]
  alpha4 = -popt4[1]
  gc1 = 1./popt1[2]
  gc2 = 1./numpy.sqrt(popt2[2])
  gc41 = 1./popt4[2]
  gc42 = 1./numpy.sqrt(popt4[3])

  icov1 = numpy.linalg.inv(pcov1)
  icov2 = numpy.linalg.inv(pcov2)
  icov3 = numpy.linalg.inv(pcov3)
  icov4 = numpy.linalg.inv(pcov4)

  # change in alpha that could double chi^2 
  # (hence a change that we can't distinguish from the random 
  #  variability that leads to chi^2>0 in the first place)
  nu = gT.size - 3
  alphaErr1 = numpy.sqrt(nu / icov1[1,1])
  alphaErr2 = numpy.sqrt(nu / icov2[1,1])
  alphaErr3 = numpy.sqrt(nu / icov3[1,1])
  alphaErr4 = numpy.sqrt(nu / icov4[1,1])

  if 1: #{
    dpopt1 = 0*popt1
    dpopt1[1] += alphaErr1
    dchi1 = chiSqr(lnPowLawExp, gT, lndNdgT, popt1+dpopt1, 
                   sigmas = lndNdgStdT)[0] - chi1
    dpopt2 = 0*popt2
    dpopt2[1] += alphaErr2
    dchi2 = chiSqr(lnPowLawDblExp, gT, lndNdgT, popt2 + dpopt2, 
                  sigmas = lndNdgStdT)[0] - chi2
    print popt1, popt1+dpopt1
    print "chi1 = %.2g, dchi1(alphaErr1) = %.2g; chi2 = %.2g, dchi2(alphaErr2) = %.2g" % (
      chi1, dchi1, chi2, dchi2)
  #}
  
  # change in gc that could double chi^2 
  #   for lndNdg = stuff - b g^c
  #  bErr = sqrt(chiSqr / icov[2,2])
  #  and since gc = (1./b)^(1/c),
  #  gcErr = 1/[c b^(1+1/c)] bErr  (assuming c is constant)
  gcErr1 = 1./(1. * popt1[2]**2) * numpy.sqrt(nu / icov1[2,2])
  gcErr2 = 1./(2. * popt2[2]**1.5) * numpy.sqrt(nu / icov2[2,2])
  nu -= 1
  #gcErr3 = 1./(2. * popt3[2]**(1.+1./popt3[3])) * numpy.sqrt(chi3 / icov3[2,2])
  # have to add dgc/dc term for gcErr3 if we want it
  gc1Err4 = 1./(1. * popt4[2]**2) * numpy.sqrt(nu / icov4[2,2])
  gc2Err4 = 1./(2. * popt4[3]**1.5) * numpy.sqrt(nu / icov4[3,3])
  nu += 1

  oneIsBest = (chiTail1 <= chiTail2)
  if lndNdgStd is None:
    #w1 = chi2 / (chi1 + chi2)
    #w2 = chi1 / (chi1 + chi2)
    w1 = chiTail2 / (chiTail1 + chiTail2)
    w2 = chiTail1 / (chiTail1 + chiTail2)
  else:
    w1 = q1 / (q1+q2)
    w2 = q2 / (q1+q2)
  #alpha = w1*alpha1 + w2*alpha2
  #gc = gc1**w1 * gc2**w2
  alpha = alpha1 if oneIsBest else alpha2
  gc = gc1 if oneIsBest else gc2
  fbest = f1 if oneIsBest else f2

  chi = chi1 if oneIsBest else chi2
  chiTail = chiTail1 if oneIsBest else chiTail2
  
  # Measure best alpha
  if fitToLowerGamma: #{
    # alpha is belong to the curve with lowest gMinN
    # If there's a tie, take the one with the best tail fit
    gMinAs = numpy.array([gMin1, gMin2, gMin4])
    print "gMinAs =", gMinAs
    best = (gMinAs == gMinAs.min())
    iBest = numpy.where(best)[0]
    if len(iBest) == 1: #{
      iBest = iBest[0]
    else: #}{
      chiTails = numpy.array([chiTail1, chiTail2, chiTail4])
      print "chiTails =", chiTails
      chiTails = chiTails[iBest]
      icBest = chiTails.argmin()
      iBest = iBest[icBest]
    #}
    iBestAlpha = iBest
    alpha = numpy.array([alpha1, alpha2, alpha4])[iBestAlpha]
    print "iBestAlpha =", iBestAlpha, ", alpha = %.3g" % alpha
  #}

  # Using best alpha, fit for gc1 and gc2
  print "alpha=", alpha
  popt5, pcov5 = curveFit4(gT, lndNdgT, alpha=alpha)
  chi5,q5 = chiSqr(lnPowLawExpDblExp, gT, lndNdgT, popt5, sigmas = lndNdgStdT)
  chiTail5,qt5 = chiSqr(lnPowLawExpDblExp, gm, lndNdgm, popt5, sigmas = lndNdgStdm)
  gc51 = 1./popt5[2]
  gc52 = 1./numpy.sqrt(popt5[3])



  
  if 0: #{
    print "Tail with max-shot-noise of %.2g %% extends to g = %.2g and dN/dg = %.2g" % (
      maxShotErr * 100, g[igMax2-1], dNdg[igMax2-1])
    print "weights", w1, w2
    print "p1    =", p1
    print "popt1 =", popt1
    print "popt2 =", popt2
    print "popt3 =", popt3
    print "popt4 =", popt4
    print "popt5 =", popt5
    
    print "      chi1 = %.2g, chi2 = %.2g, chi4 = %.2g, chi5 = %.2g" % (chi1, chi2, chi4, chi5)
    #print "        q1 = %.2g,   q2 = %.2g,   q3 = %.2g" % (q1, q2, q3)
    print "tail: chi1 = %.2g, chi2 = %.2g, chi4 = %.2g, chi5 = %.2g" % (
      chiTail1, chiTail2, chiTail4, chiTail5)
    #print "tail:   q1 = %.2g,   q2 = %.2g,   q3 = %.2g" % (qt1, qt2, qt3)

    #print "alpha = %.3g, gc = %.3g, oddsOfExp = %.2g" % (
    #  alpha, gc, w1/w2)
    print "  alpha = %.3g, %.3g, %.3g, %.3g" % (
      alpha1, alpha2, alpha4, alpha)
    print "1 std error in alpha and gc"
    print "   aErr1 = %.2g (%.2g%%),  aErr2 = %.2g (%.2g%%), aErr4 = %.2g (%.2g%%)" % (
      alphaErr1, abs(alphaErr1/alpha1)*100, alphaErr2, abs(alphaErr2/alpha2)*100,
      alphaErr4, abs(alphaErr4/alpha4)*100)
    print "  gc1 (1, 4, 5) = %.3g, %.3g, %.3g" % (gc1, gc41, gc51)
    print "  gc2 (2, 4, 5) = %.3g, %.3g, %.3g" % (gc2, gc42, gc52)
    print "  gcErr1 = %.2g (%.2g%%), gcErr2 = %.2g (%.2g%%), gc1Err4 = %.2g (%.2g%%) gc2Err4 = %.2g (%.2g%%)" % (
      gcErr1, abs(gcErr1/gc1)*100, gcErr2, abs(gcErr2/gc2)*100, 
      gc1Err4, abs(gc1Err4)/gc41*100, gc2Err4, abs(gc2Err4)/gc42*100)
  #}


  if 0: #{
    import pylab
    gp = 1.
    plotCmd = pylab.loglog
    dNdgggp = dNdg * g**gp
    ym = (dNdgggp[dNdgggp>0]).min()

    def doPlot(f, c, g0, g0Err, label): #{
      y = numpy.exp(f(g))
      y *= g**gp
      yr = (y > ym)
      y = y[yr]
      gr = g[yr]
      alph1 = 0.5
      alph2 = 0.5
      lw = 1
      plotCmd(gr[:igMin], y[:igMin], "--" + c, alpha = alph2, linewidth=lw)
      plotCmd(gr[igMin:igMax], y[igMin:igMax], "-" + c, alpha = alph1, label = label, linewidth=lw)
      plotCmd(gr[igMax:], y[igMax:], "--" + c, alpha = alph2, linewidth=lw)
      if isinstance(g0, tuple):
        g01 = [max(1, min(g0[0]+i*g0Err[0], gr[-1])) for i in (-1,0,1)]
        g02 = [max(1, min(g0[1]+i*g0Err[1], gr[-1])) for i in (-1,0,1)]
        v = numpy.exp(f(g01[1]))*g01[1]**gp
        plotCmd(g01, [v]*3, "o" + c, alpha = alph2,
          mfc = 'none', markersize=8)
        v = numpy.exp(f(g02[1]))*g02[1]**gp
        plotCmd(g02, [v]*3, "o" + c, alpha = alph2,
          mfc = 'none', markersize=12)
      else:
        v = numpy.exp(f(g0))*g0**gp
        plotCmd([g0-g0Err,g0, g0+g0Err], [v]*3, "o" + c, alpha = alph2)
        #print "g0 = %g +/- %g" % (g0,g0Err)
    #}
    doPlot(f1, 'b', gc1, gcErr1, "1")
    doPlot(f2, 'g', gc2, gcErr2, "2")
    #doPlot(f3, 'y', (1./popt3[2])**(1./popt3[3]), 0, "%.2g" % popt3[3])
    doPlot(f5, 'c', (gc51, gc52), (gc1Err4,gc2Err4), "b5")
    plotCmd(g, dNdg*g**gp, '-m')
    #plotCmd(gc, numpy.exp(fbest(gc))*gc**gp, '*', mfc='none', mec='c',
    #  markersize=15, alpha=0.5)
    #plotCmd(g, stdRel, 'y-', label = 'sig')
    pylab.legend(loc='best')
    title = r"$\alpha=$%.3g, %.3g, %.3g" % (-popt1[1], -popt2[1], -popt5[1])
    title += r" $\chi^2=$%.2g, %.2g, %.2g (tail)" % (
      chiTail1, chiTail2, chiTail5)
    title += "\n"
    title += r"$\chi^2=$%.2g, %.2g, %.2g" % (chi1, chi2, chi5)
    #title += r" $\bar{\alpha}=$%.3g" % alpha
    pylab.ylabel(r"$\gamma^{%.2g}$" % gp)
    pylab.title(title)
    pylab.subplots_adjust(top=0.86)
    pylab.show()
    raise RuntimeError, "stopped after debugging plot"
  #}

  # 3rd is the "odds" that the fit is exp (compared to dbl-exp)
  return (alpha, gc, chiTail2/chiTail1, chi, chiTail,
    [popt1, popt2, popt3], [f1, f2, f3],
    [chi1,chi2,chi3], [chiTail1, chiTail2, chiTail3])
#}

def fit6(gEdges, dNdg, dMacro, stdRel,
  arbitrariness = {"gMin":1., "gMax":1., "relVar":1.}): #{
  """
  fit power-law and cutoff to g > gMin only
  """
  # need a value of gamma that's way larger than any we'll encounter
  gcHuge = 1./2e-16 # the square of this must not overflow
  # rough measure of allowed variation
  relVar = 0.02 * arbitrariness["relVar"]
  if stdRel is None:
    stdRel = estimateDistError(dNdg, dMacro, gEdges)

  g = 0.5*(gEdges[1:] + gEdges[:-1])
  lndNdg = mathPlus.logWithZeros(dNdg, zeroTreatment = ("setBelowLowest", 1))

  # First, find truncated spectrum 
  minPts = 5
  (gMin, gMax, igMin, igMax) = gMinMaxForTailFit(gEdges, 
    dNdg*numpy.diff(gEdges), arbitrariness = arbitrariness, minPts = minPts)
  print "fit6: Using gMin = %.3g, gMax = %.3g for fitting" % (gMin, gMax), ", igMin igMax:", igMin, igMax

  # Find where shot noise in tail goes above relVar
  maxShotErr = relVar
  igMaxTailWhere = numpy.where(dMacro[igMin:] < 1./maxShotErr**2)[0]
  if len(igMaxTailWhere) == 0:
    igMaxTail = dNdg.size
  else:
    igMaxTail = igMin + igMaxTailWhere[0]
    if igMaxTail < igMax:
      igMaxTail = min(igMax+1, len(g))

  #########################
  # If we don't have enough points, give up by returning reasonable result
  #########################
  if (igMaxTail - igMin < minPts) or not (dNdg[igMin:igMaxTail] > 0).all():
    #res = (alphaBest, gc1best, gc2best)
    res = (1., 1.,1.)
    gEdges = 2.**numpy.arange(0.,2.,0.125)
    g = 0.5*(gEdges[1:]+gEdges[:-1])
    dNdg = g**(-3.)*numpy.exp(-3.*g)
    dMacro = 0*g + 100.
    stdRel = None
    return fit6(gEdges, dNdg, dMacro, stdRel)




  # Fit power law + exp
  def lnPowLawExp(x, lnC, a, b): 
    # fit C x^-a exp(-b x)
    return lnC - a * numpy.log(x) - b*x 
  # Fit power law + double-exp
  def lnPowLawDblExp(x, lnC, a, b): 
    # fit C x^-a exp(-b x**2)
    return lnC - a * numpy.log(x) - b*x**2
  def lnPowLawExpDblExp(x, lnC, a, b1, b2): 
    # fit C x^-a exp(-b1 x - b2 x^2)
    return lnC - a * numpy.log(x) - x*(b1 + b2*x)

  def avgChiSqr(fn, xs, ys, p0, sigma = None): #{
    N = len(xs)
    if sigma is None:
      sigma = 1.
    yFits = fn(xs, *p0)
    res = ( ((ys - yFits)/ sigma)**2 ).sum()
    nu = N - len(p0)
    if nu == 0:
      raise RuntimeError, "nu = 0"
    # probability, given sigmas, that chi^2 > the measured value
    #Q = scipy.special.gammaincc(0.5 * nu , 0.5 * res)
    # find average to allow some comparison
    res /= nu
    return res #(res, Q)
  #}
  
  def getGc(b, oneOrTwo): #{
     if oneOrTwo not in [1,2]:
       raise ValueError, "oneOrTwo must be 1 or 2"
     if oneOrTwo == 1:
       if abs(b) < 1./gcHuge:
         res = gcHuge 
         if b != 0:
           res *= numpy.sign(b)
       else:
         res = 1./b 
     elif oneOrTwo == 2:
       if numpy.sqrt(abs(b)) < 1./gcHuge:
         res = gcHuge 
       else:
         res = 1./numpy.sqrt(abs(b)) 
       if b != 0:
         res *= numpy.sign(b)
     return res
  #}
    
  # Find rough initial fit to use in fitAll
  (C0, a0, b0, pcovb) = mathPlus.fitPowerLawExp(g[igMin:igMax], 
    dNdg[igMin:igMax], ySigmas = None, fit = "log")
  lnC0 = numpy.log(C0)
  a0 = -a0 
  b0 = -b0

  def fitAll(gs, lnfs, sigma = None, alpha = None): #{
    """
    if alpha is given, fix alpha (power-law index) to that value
    """
    a0f = a0 if alpha is None else alpha
    p1 = [lnC0, a0f, b0]
    p2 = [lnC0, a0f, numpy.sqrt(abs(b0))]
    cpi = [] if alpha is None else [1]

    (popt1, pcov1) = optPlus.curveFitWithFixedParams(scipy.optimize.curve_fit, 
      lnPowLawExp, gs, lnfs, cpi, p0 = p1, sigma = sigma, maxfev=defMaxFev)
    (popt2, pcov2) = optPlus.curveFitWithFixedParams(scipy.optimize.curve_fit, 
      lnPowLawDblExp, gs, lnfs, cpi, p0 = p2, sigma = sigma, maxfev=defMaxFev)

    if 1: #{
      if popt1[2] < 0:
        naRange = None if alpha is None else [alpha, alpha]
        (popt1, pcov1) = optPlus.curveFitWithBoundedParams(scipy.optimize.curve_fit, 
          lnPowLawExp, gs, lnfs, 
          paramMags = [1., 1., 1e-4],
          paramBounds = [None, naRange, [0.,1.]],
          p0=p1, maxfev=defMaxFev)
      if popt2[2] < 0:
        naRange = None if alpha is None else [alpha, alpha]
        (popt2, pcov2) = optPlus.curveFitWithBoundedParams(scipy.optimize.curve_fit, 
          lnPowLawExp, gs, lnfs, 
          paramMags = [1., 1., 1e-8],
          paramBounds = [None, naRange, [0.,1.]],
          p0=p2, maxfev=defMaxFev)
    #}
    
    p0b1 = list(popt1) + [0.]
    p0b2 = list(popt2[:-1]) + [0.,popt2[-1]]
    starts = [
      ["new", p0b1],
      ["new", p0b2],
      ["best"]
    ]
    if alpha is not None:
      for s in starts:
        s.append((1, alpha))

    (poptb, pcovb) = optPlus.curveFitWithBoundedParamsMultipleStarts(
      scipy.optimize.curve_fit, 
      lnPowLawExpDblExp, gs, lnfs, 
      paramMags = [1., 1., 1e-4, 1e-8],
      paramBounds = [None, None, [0.,1.],[0.,1.]],
      starts = starts,
      p0 = p0b1, sigma = None, maxfev=defMaxFev)[0]
    
    aChiSqr1 = avgChiSqr(lnPowLawExp, gs, lnfs, popt1, sigma = sigma)
    aChiSqr2 = avgChiSqr(lnPowLawDblExp, gs, lnfs, popt2, sigma = sigma)
    aChiSqr3 = avgChiSqr(lnPowLawExpDblExp, gs, lnfs, poptb, sigma = sigma)

    res = ([popt1, popt2, poptb], 
           [pcov1, pcov2, pcovb], 
      numpy.array([[aChiSqr1, aChiSqr2, aChiSqr3],
                   [popt1[1], popt2[1], poptb[1]], # alpha
                   [popt1[2], 0., poptb[2]], # 1./gc1
                   [0., popt2[2], poptb[3]]]) ) # 1./sqrt(gc2)

    return res
  #}


  def extendFit(relVar, gs, lnfs, igStart, sigma = None, extendDir = -1,
    alpha = None): #{
    """
    igStart should be the (lowest/highest) index (for extendDir = -1/1)
    that should definitely be included (e.g., even if no extension
    occurs).  If extendDir = 1, gs[igStart] is included.
    """
    
    if abs(extendDir) != 1:
      raise ValueError, "extendDir must be either -1 (down in g) or +1 (up in g)"

    print "gs.shape =", gs.shape, " igstart =", igStart
    if extendDir == -1:
      (popts0, pcovs0, resra0) = fitAll(gs[igStart:], lnfs[igStart:], 
        sigma = sigma, alpha = alpha)
    else:
      (popts0, pcovs0, resra0) = fitAll(gs[:igStart+1], lnfs[:igStart+1], 
        sigma = sigma, alpha = alpha)
    chi0 = resra0[0].min()
    
    def getCmp(ig): #{
      if extendDir == -1:
        # Shorter gamma-ranges mean a change in dNdg at one end
        # changes alpha (the slope) more.
        # Try to make so that this returns the amount that one endpoint
        # could move such that the slope would change by relVar
        res = numpy.log(gs[-1]/gs[igStart])
      else:
        res = 1.
      res *= relVar
      return res
    #}
    def getChiCmp(cmpFactor): #{
      #return cmpFactor
      return numpy.sqrt(cmpFactor**2 + chi0**2)
    #}
    igRange = range(igStart) if extendDir < 0 else range(len(gs)-1, igStart-1,-1)
    for ig in igRange: #{
      if extendDir == -1:
        igFit1 = ig
        igFit2 = -1
        ig1 = ig
        ig2 = igStart
      else: 
        igFit1 = 0
        igFit2 = ig+1
        ig1 = igStart
        ig2 = ig+1
      sigma = None if sigma is None else sigma[igFit1, igFit2]
      (popts, pcovs, resra) = fitAll(gs[igFit1:igFit2], lnfs[igFit1:igFit2], 
        sigma = sigma)
      if extendDir == -1:
        dev1 = abs(lnPowLawExp(      gs[ig1:ig2], *popts[0]) - lnfs[ig1:ig2]).max()
        dev2 = abs(lnPowLawDblExp(   gs[ig1:ig2], *popts[1]) - lnfs[ig1:ig2]).max()
        dev3 = abs(lnPowLawExpDblExp(gs[ig1:ig2], *popts[2]) - lnfs[ig1:ig2]).max()
        devs = [dev1, dev2, dev3]
      else:
        devs = numpy.zeros((3,))
      chis = numpy.sqrt(resra[0])
      cmpFactor = getCmp(ig)
      chiCmpFactor = getChiCmp(cmpFactor)
      goodErr = numpy.array([numpy.isfinite(pc).all() for pc in pcovs])
      good = numpy.logical_and(numpy.logical_and(
                               goodErr,
                               devs <= cmpFactor),
                               chis <= chiCmpFactor)
      if good.any():
        break
    #}
    if good.sum() == 1: #{ Go with the fit that worked to lowest gamma
      iaBest = numpy.where(good)[0][0]
    else: #}{ if more than one (or none, ig=igStart), break the tie with lowest chi
      iaBest = chis.argmin()
      if not numpy.isfinite(pcovs[iaBest]).all():
        msg = "Best fit had inf pcov ?!"
        raise RuntimeError, msg
    #}

    igExtGamma = ig
    gExtGamma = gs[igExtGamma]

    if extendDir == -1:
      alphaBest = resra[1][iaBest]
      res = (alphaBest, igExtGamma, gExtGamma, popts, pcovs, resra, good, iaBest)
    else:
      gc1best = getGc(resra[2][iaBest], 1)
      gc2best = getGc(resra[3][iaBest], 2)
      # give exclusive upper bound, hence igExtGamma+1
      res = (gc1best, gc2best, igExtGamma+1, gExtGamma, popts, pcovs, resra, good, iaBest)
    return res
  #}
  
  (alphaBest, igLowGamma, gLowGamma, popts, pcovs, resra, good, iaBest
    ) = extendFit(relVar, g[:igMax], lndNdg[:igMax], igMin,
      sigma = None)
  
  if 0: #{
    (poptsA, pcovsA, resraA) = fitAll(
      g[igLowGamma:igMax], lndNdg[igLowGamma:igMax], sigma = None,
      alpha = alphaBest)

    (popt1, popt2, poptb) = poptsA
    
    gc1 = getGc(popt1[2], 1)
    gc2 = getGc(popt2[2], 2)
    if poptb[2] < 0. or poptb[3] < 0.:
      print "Warning: poptb[2,3] should be >= 0, but poptb[2:] =", poptb[2:]
    if poptb[2] <= 0.:
      poptb[2] = 1./gcHuge
    if poptb[3] <= 0.:
      poptb[3] = 1./gcHuge**2
    gcb1 = getGc(poptb[2], 1)
    gcb2 = getGc(poptb[3], 2) 

    aChiSqrs = resraA[0]
    giBest = aChiSqrs.argmin()
    if aChiSqrs[2] == aChiSqrs[giBest]:
      # go with exp/dbl-exp fit if there's a tie
      giBest =2
    if giBest == 0:
      gc1best = gc1
      gc2best = gcHuge
    elif giBest == 1:
      gc1best = gcHuge
      gc2best = gc2
    else:
      gc1Best = gcb1
      gc2Best = gcb2

  #}

  # Now fit with fixed alpha, extend to higher gamma
  print "igLoG, igMin, igMax, igMaxTail", igLowGamma, igMin, igMax, igMaxTail
  (gc1best, gc2best, igHiGamma, gHiGamma, poptsA, pcovsA, resraA, good, igBest
    ) = extendFit(relVar, g[igLowGamma:igMaxTail], 
      lndNdg[igLowGamma:igMaxTail], igMax-1 - igLowGamma, sigma = None, 
      extendDir = 1, alpha = alphaBest)
  igHiGamma += igLowGamma
    
  gc1 = getGc(poptsA[0][2], 1)
  gc2 = getGc(poptsA[1][2], 2)
  gcb1 = getGc(poptsA[2][2], 1)
  gcb2 = getGc(poptsA[2][3], 2)
  
  (popt1, popt2, poptb) = poptsA

  def f1(x):
    res = lnPowLawExp(x, *poptsA[0])
    return res
  def f2(x):
    res = lnPowLawDblExp(x, *poptsA[1])
    return res
  def fb(x):
    res = lnPowLawExpDblExp(x, *poptsA[2])
    return res
  fbest = (f1, f2, fb)[igBest]

  if 1: #{
    # pcovsA are singular because alpha was fixed
    def invpcov(m): #{
      m = numpy.array(m)
      (r,c) = m.shape
      rows = [[n] for n in range(r) if n != 1]
      cols = [n for n in range(c) if n != 1]
      mInv = 0*m
      mInv[rows, cols] = numpy.linalg.inv(m[rows, cols])
      return mInv
    #}
    icov1 = invpcov(pcovsA[0])
    icov2 = invpcov(pcovsA[1])
    icovb = invpcov(pcovsA[2])

    # change in alpha that could double chi^2 
    # (hence a change that we can't distinguish from the random 
    #  variability that leads to chi^2>0 in the first place)
    nu = igMax - igLowGamma - 3
    #alphaErr1 = numpy.sqrt(nu / icov1[1,1])
    #alphaErr2 = numpy.sqrt(nu / icov2[1,1])
    nu += 1
    #alphaErrb = numpy.sqrt(nu / icovb[1,1])
    nu -= 1

    if 0: #{ # test that we've calculated error correctly
      dpopt1 = 0*popt1
      dpopt1[1] += alphaErr1
      dchi1 = chiSqr(lnPowLawExp, gT, lndNdgT, popt1+dpopt1, 
                     sigmas = lndNdgStdT)[0] - chi1
      dpopt2 = 0*popt2
      dpopt2[1] += alphaErr2
      dchi2 = chiSqr(lnPowLawDblExp, gT, lndNdgT, popt2 + dpopt2, 
                    sigmas = lndNdgStdT)[0] - chi2
      #print popt1, popt1+dpopt1
      #print "chi1 = %.2g, dchi1(alphaErr1) = %.2g; chi2 = %.2g, dchi2(alphaErr2) = %.2g" % (
      #  chi1, dchi1, chi2, dchi2)
    #}
    
    # change in gc that could double chi^2 
    #   for lndNdg = stuff - b g^c
    #  bErr = sqrt(chiSqr / icov[2,2])
    #  and since gc = (1./b)^(1/c),
    #  gcErr = 1/[c b^(1+1/c)] bErr  (assuming c is constant)
    if popt1[2] <= 0.:
      gcErr1 = gcHuge
    else:
      gcErr1 = 1./(1. * popt1[2]**2) * numpy.sqrt(nu / icov1[2,2])
    if popt2[2] <= 0.:
      gcErr2 = gcHuge
    else:
      gcErr2 = 1./(2. * popt2[2]**1.5) * numpy.sqrt(nu / icov2[2,2])
    nu -= 1
    if poptb[2] <= 0.:
      gc1Errb = gcHuge
    else:
      gc1Errb = 1./(1. * poptb[2]**2) * numpy.sqrt(nu / icovb[2,2])
    if poptb[3] <= 0.:
      gc2Errb = gcHuge
    else:
      gc2Errb = 1./(2. * poptb[3]**1.5) * numpy.sqrt(nu / icovb[3,3])
    nu += 1
  #}

  rmsChis = numpy.sqrt(resraA[0])
  rmsChiBest = rmsChis[igBest]
  res = (alphaBest, gc1best, gc2best, fbest, 
         [gLowGamma, gHiGamma], rmsChiBest, [gc1, gc2, gcb1, gcb2],
         igBest, (f1, f2, fb), poptsA, pcovsA, rmsChis)
  
  if 0: #{
    print "popt1 =", popt1
    print "popt2 =", popt2
    print "poptb =", poptb
    print " gMin -> gMax = %.2g -> %.2g" % (gMin, gMax)
    print "best alpha = %.3g, fit with %.2g%% relVar from g = %.2g -> %.2g" % (
      alphaBest, relVar*100, gLowGamma, gHiGamma)
    #print "     gc1 = %.2g +/- %.2g%%,  gc2 = %.2g +/- %.2g%%" % (
    #  gc1, abs(gcErr1)/gc1*100, gc2, abs(gcErr2)/gc2*100)
    #print "    gcb1 = %.2g +/- %.2g%%, gcb2 = %.2g +/- %.2g%%" % (
    #  gcb1, abs(gc1Errb)/gcb1*100, gcb2, abs(gc2Errb)/gcb2*100)
    print "best gc1 = %.2g +/- %.2g%%,  gc2 = %.2g +/- %.2g%%" % (
      gc1best, abs(gcErr1)/gc1best*100, gc2best, abs(gcErr2)/gc2best*100)
    print " chiSqrs: %.2g, %.2g, %.2g" % tuple(resraA[0])
  #}

  if 0: #{
    import pylab
    setFontSize()
    gp = 1.
    plotCmd = pylab.loglog
    dNdgggp = dNdg * g**gp
    ym = (dNdgggp[dNdgggp>0]).min()

    def ongraph(gamma):
      igamma = numpy.argmin(abs(g-gamma))
      #if gamma < g[igamma] and gamma > 0:
      #  igamma -= 1
      res = max(ym, dNdgggp[igamma])
      return (min(gamma, g[-1]), res)
    def doPlot(f, c, g0, g0Err, label, linestyle='--'): #{
      y = numpy.exp(f(g))
      y *= g**gp
      yr = (y > ym)
      y = y[yr]
      gr = g[yr]
      alph1 = 0.5
      alph2 = 0.5
      lw = 4
      print "grLo =", gr[:igMin]
      print "grMid =", gr[igMin:igMax]
      print "grHi =", gr[igMax:]
      print "fLo =", y[:igMin]
      print "fMid =", y[igMin:igMax]
      print "fHi =", y[igMax:]
      
      plotCmd(gr[:igMin], y[:igMin], linestyle + c, alpha = alph2, linewidth=lw)
      plotCmd(gr[igMin:igMax], y[igMin:igMax], "-" + c, alpha = alph1, linewidth=lw)
      plotCmd(gr[igMax:], y[igMax:], linestyle + c, alpha = alph2, linewidth=lw,
        label=label)
      if 0: #{
        if isinstance(g0, tuple):
          g01 = [max(1, min(g0[0]+i*g0Err[0], gr[-1])) for i in (-1,0,1)]
          g02 = [max(1, min(g0[1]+i*g0Err[1], gr[-1])) for i in (-1,0,1)]
          v = numpy.exp(f(g01[1]))*g01[1]**gp
          plotCmd(g01, [v]*3, "o" + c, alpha = alph2,
            mfc = 'none', markersize=8)
          v = numpy.exp(f(g02[1]))*g02[1]**gp
          plotCmd(g02, [v]*3, "o" + c, alpha = alph2,
            mfc = 'none', markersize=12)
        else: 
          v = numpy.exp(f(g0))*g0**gp
          plotCmd([g0-g0Err,g0, g0+g0Err], [v]*3, "o" + c, alpha = alph2)
          #print "g0 = %g +/- %g" % (g0,g0Err)
      #}
    #}
    doPlot(f1, 'b', gc1, gcErr1, "exponential")
    doPlot(f2, 'g', gc2, gcErr2, "super-exp", linestyle='-.')
    #doPlot(fb, 'c', (gcb1, gcb2), (gc1Errb,gc2Errb), "b")
    plotCmd(g, dNdg*g**gp, '-m', lw=2)
    print "g =", g
    print "dNdg * g^%i =" % (gp,), (dNdg*g**gp)
    #plotCmd(*ongraph(gc1best), marker='s', mec='m', alpha=0.25, markersize=8, mfc='none')
    #plotCmd(*ongraph(gc2best), marker='s', mec='m', alpha=0.25, markersize=16, mfc='none')
    pylab.legend(loc='best')
    title = r"$\alpha=$%.3g" % (alphaBest,)
    title += "\n"
    title += r"$\chi^2=$%.2g, %.2g, %.2g" % tuple(resraA[0])
    if gp == 1:
      pylab.ylabel(r"$\gamma f(\gamma)$")
    else:
      pylab.ylabel(r"$\gamma^{%.2g}f(\gamma)$" % gp)
    #pylab.title(title)
    pylab.xlabel(r"$\gamma$", fontsize=defFontSize+2)
    #pylab.subplots_adjust(top=0.86)
    pylab.subplots_adjust(bottom=0.15, top = 0.95, left=0.15)
    pylab.show()
    raise RuntimeError, "stopped after debugging plot"
  #}

  return res
#}

def findLongestConsecutiveTrue(boolRa, xs): #{
  """
  boolRa is a 1D boolean array;
  xs is an array with 1 more element that boolRa, with
  boolRa[i] located between xs[i] and xs[i+1].
  The length of a consecutive run of True in boolRa is the difference 
  between the xs surrounded the run.

  """
  intRa = numpy.hstack(([0], numpy.int32(boolRa), [0]))
  if not intRa.any(): #{
    res = (0, 0, 0, 0, 0)
  else: #}{
    dif = numpy.diff(intRa)
    starts = numpy.where(dif==1)[0]
    stops = numpy.where(dif==-1)[0]
    #print boolRa
    #print xs
    #print starts, stops
    lengths = xs[stops] - xs[starts]
    #print lengths
    i = lengths.argmax()
    maxLen = lengths[i]
    res = (maxLen, starts[i], stops[i], xs[starts[i]], xs[stops[i]])
    #print res
  #}
  return res
#}

def findLongestPowerLawIndex(x, dNdx, indexSlop = 0.05,
  numSmooths = 0): #{
  """
  x and dNdx are arrays of equal length: dNdx[i] = dN/dx (x[i])
  such that dNdx[i] * dx = the number of things within [x[i], x[i]+dx]
    for sufficiently small dx.
  Note that dNdx[i] is not the number of things within
    [x[i], x[i+1]].

  Tries to find the longest stretch [log(x1), log(x2)], such that
  d log(dNdx) / d log(x) = a +/- indexSlop within the interval.

  Will smooth the d log(dNdx) / 
  
  Here, log is log10.
  """
  logdNdx = numpy.log10(dNdx)
  logx = numpy.log10(x)
  dlogx = numpy.diff(logx)
  slope = arrayPlus.smooth1d(numpy.diff(logdNdx) / dlogx, numSmooths)
  # center difference
  newLogx = 0.5*(logx[:-1] + logx[1:])
  newx = 10.**newLogx
  
  # search for long stretches
  slopeMin = slope.min()
  slopeMax = slope.max()
  # identify likely indices
  numBins = max(1, (slopeMax - slopeMin) / indexSlop)
  #numpy.histogram(slope, numBins, [slopeMin, slopeMax], weights = dlogx)
  sBest = slopeMin
  sBestLen = 0.
  dIndex = indexSlop/5.
  sStart = round(slopeMin/dIndex) * dIndex
  sBestxStart = 0.
  if 0: #{
    pylab.figure()
    for ns in [0,2,4,16,64,256]:
      pylab.semilogx(newx, arrayPlus.smooth1d(slope, ns), label="%i" % ns)
    pylab.legend(loc='best')
    pylab.show()
  #}
  for s in numpy.arange(sStart, slopeMax, dIndex): #{
    (sLen, iStart, iStop, logxStart, logxStop) = findLongestConsecutiveTrue(
      abs(slope - s) <= indexSlop, logx)
    #print "Longest section with slope %.2f: %.3g -> %.3g, ratio = %.3g" % (s, 
    #  10.**logxStart, 10.**logxStop, 10.**(logxStop-logxStart))
    if sLen >= sBestLen:
      sBestLen = sLen
      sBest = s
      sBestxStart = 10.**logxStart
  #}
  return (sBest, sBestLen, sBestxStart)
#}



def getMagEnergyLoss(sDict): #{
  simName = sDict["simName"]
  bu = vorpalUtil.getHistory(simName, "energyMagUp")
  bd = vorpalUtil.getHistory(simName, "energyMagDn")
  timesAndSteps = vorpalUtil.getHistory(simName, "energyCalcTimes")
  b = bu + bd
  b0 = (bu+bd)[0]
  bMin = b.min()
  maxDrop = 1. - bMin / b0
  bThresh5pct = bMin + 0.05*(b0-bMin)
  bLow = b[b <= bThresh5pct]
  bSteady = bLow.mean()
  #print "Bdrop: max = %.4g%%, to steady = %.4g%%" % (maxDrop*100, 
  #  (1.-bSteady/b0)*100.)
  #print ",%.4g, %.4g" % (maxDrop*100, (1.-bSteady/b0)*100.)
  return (b0, bSteady, bLow)
#}

#getMagEnergyLoss({"simName":"relRecon2p"})


def addEnergyGrowthAndEndOfReconnectionTime(sDict): #{
  simName = sDict["simName"]
  omegac = sDict["omegac"]
  bu = vorpalUtil.getHistory(simName, "energyMagUp")
  bd = vorpalUtil.getHistory(simName, "energyMagDn")
  timesAndSteps = vorpalUtil.getHistory(simName, "energyCalcTimes")
  numSteps = timesAndSteps.shape[0]
  tLast = timesAndSteps[-1][0]
  b = bu + bd
  b0 = (bu+bd)[0]
  bMin = b.min()
  dt = timesAndSteps[1,0] - timesAndSteps[0,0]
  L = sDict["LY_TOT"]
  boxCrossingTime = L/c
  nSmoothLen = int(round(boxCrossingTime/dt))
  #print nSmoothLen
  #n2 = int(round(numpy.log(nSmoothLen)/numpy.log(2.)))
  bSmooth = arrayPlus.smooth1d(b, nSmoothLen)
  bSmoothMin = bSmooth.min()
  #dropFrac = getEnergyDisFrac(sDict["sigmae"], sDict["LY_TOT"],
  #  sDict["rho0"])
  dropFrac = 1. - bSmoothMin/b0
  dropFracFudge = dropFrac * 0.97
  maxDrop = 1. - bMin / b0
  if maxDrop < dropFrac:
    raise ValueError, "Maximum percent drop = %g %% < %g %%" % (
      maxDrop*100, dropFrac*100)
  bThresh = b0 * (1. - dropFracFudge)
  iDone = numpy.where(b <= bThresh)[0][0]
  (t,n) = timesAndSteps[iDone]
  iDone2 = numpy.where(timesAndSteps[:,0] < t+boxCrossingTime)[0][-1]
  (t2,n2) = timesAndSteps[iDone2]
  if 1: #{
    # estimate actual dissipated fraction in steady state end
    iNearEnd = max(iDone,
      numpy.where(timesAndSteps[:,0] > tLast-boxCrossingTime)[0][0])
    endBsmooth = bSmooth[iNearEnd:]
    bSteady = 0.5 * (endBsmooth.max() + endBsmooth.min())
    dropFracBestEst = 1. - bSteady/b0
    sDict["magEnergyDropFrac"] = dropFracBestEst
    #print 'dropFrac = %g, %g -> %g' % (dropFracBestEst, b0, bSteady)
  #}
  if 0: #{
    import pylab
    import sys
    import os
    pylab.plot(omegac*timesAndSteps[:,0], b, 'o', mfc = 'none', alpha=0.2)
    pylab.plot(omegac*timesAndSteps[:,0], bSmooth, alpha=0.3)
    #pylab.plot(omegac*timesAndSteps[iNearEnd:,0], bSmooth[iNearEnd:])
    pylab.plot(omegac*timesAndSteps[iDone:iDone2:iDone2-iDone-1,0], 
      [bThresh]*2, '-x')
    pylab.plot(omegac*timesAndSteps[iNearEnd:numSteps:numSteps-iNearEnd-1,0], 
      [bSteady]*2, ':s', mfc = 'none')
    pylab.title(os.getcwd())
    #pylab.plot(omegac*timesAndSteps[:-1,0], numpy.diff(b)/b0*boxCrossingTime/dt, 'o')
    #pylab.plot(omegac*timesAndSteps[:-1,0], numpy.diff(bSmooth)/b0*boxCrossingTime/dt, '-')
    pylab.show()
    sys.exit()
  #}
  if 0:
    (tMax, nMax) = timesAndSteps[-1]
    print "Drop in Mag Energy = %g %% ; drop %g %% at omega_c t = %g, " % (
      maxDrop*100, dropFracFudge*100, omegac*t)
    print "   at n/nMax = %i/%i = %.2g" % (n, nMax, float(n)/nMax)
  otherEnergies = ["energyElecUp", "energyElecDn",
   "updriftIonsEnergy", "dndriftIonsEnergy",
   "updriftElectronsEnergy", "dndriftElectronsEnergy",
   "upbgIonsEnergy", "dnbgIonsEnergy",
   "upbgElectronsEnergy", "dnbgElectronsEnergy"]
  totalEnergy = bu + bd
  for hist in otherEnergies:
    totalEnergy += vorpalUtil.getHistory(simName, hist)
  relEnergyGrowth = (totalEnergy - totalEnergy[0])/totalEnergy[0]
  maxEnergyVar1 = abs(relEnergyGrowth[:iDone+1]).max()
  maxEnergyVar2 = abs(relEnergyGrowth[:iDone2+1]).max()
  sDict["tReconEnd"] = t
  sDict["tReconEndPlusCrossTime"] = t2
  sDict["nReconEnd"] = n
  sDict["nReconEndPlusCrossTime"] = n2
  sDict["energyGrowthAtReconEnd"] = relEnergyGrowth[iDone]
  sDict["energyGrowthAtReconEndPlusCrossTime"] = relEnergyGrowth[iDone2]
  sDict["energyVarByReconEnd"] = maxEnergyVar1
  sDict["energyVarByReconEndPlusCrossTime"] = maxEnergyVar2
  if t2 >= tLast:
    msg = "unfinished?"
    if "problemQ" not in sDict or sDict["problemQ"] < 1:
      sDict["problemQ"] = 1
    if t + 0.5*boxCrossingTime > tLast:
      msg = "unfinished!?"
      if sDict["problemQ"] < 2:
        sDict["problemQ"] = 2
    if "problems" not in sDict:
      sDict["problems"] = []
    sDict["problems"].append(msg)
  return (t,n, t2, n2, dropFracBestEst, relEnergyGrowth[iDone], relEnergyGrowth[iDone2], 
    maxEnergyVar1, maxEnergyVar2)

#}

def getUnreconnectedFluxVsTime(simName):
  """
  This is the unreconnected flux between the two layers; a line integral
  of Az from the minimum Az in the lower layer to the maximum Az in
  the upper layer (i.e., from x-point to x-point).
  """
  bxFlux = vorpalUtil.getHistory(simName, "bxFluxLeftXO")[:,0]
  dt = vorpalUtil.getHistoryDt(simName)
  ts = numpy.arange(len(bxFlux)) * dt
  bxFluxFn = scipy.interpolate.interp1d(ts, bxFlux, kind = "linear",
    bounds_error = False, fill_value = bxFlux[-1])
  
  # get horizontal B histories
  dnByName = "layerDnByLine"
  (dnBy, c, dnByLbs, dnByUbs) = vorpalUtil.getFieldArrayHistory(simName,
    dnByName)
  (upBy, c, upByLbs, upByUbs) = vorpalUtil.getFieldArrayHistory(simName,
    "layerUpByLine")
  byHistTimes = vorpalUtil.getHistoryTimes(simName, dnByName)[:,0]

  (ndim, numPhysCells, startCell, lowerBounds, upperBounds
    ) = vorpalUtil.getSimGridInfo(simName)
  dxs = (upperBounds-lowerBounds) / numPhysCells
  if ndim == 2:
    (dx, dy) = dxs
    dz = 1.
  elif ndim == 3:
    (dx, dy, dz) = dxs

  # Since Bx < 0 between layers, bxFlux is negative
  # bxFlux is Bx integrated in the upward (+y) direction
  # convert to Az, integrating in the right (+x) direction from left side
  dnAz = dnBy[...,0].cumsum(axis=1) * (-dx*dz)
  upAz = upBy[...,0].cumsum(axis=1) * (-dx*dz)
  
  # get x/i coordinates of bxFluxLeftXO
  # - assume it's 1/4 of way across width
  iFluxLeft = dnByUbs[0]/4

  # Now, we need to normalize dnAz and upAz so that they are zero at
  # iFluxLeft:
  dnAzFluxLeft = dnAz[:,iFluxLeft].copy()
  upAzFluxLeft = upAz[:,iFluxLeft].copy()
  dnAz -= dnAz[:, iFluxLeft, numpy.newaxis].copy()
  upAz -= upAz[:,iFluxLeft, numpy.newaxis].copy()
  # E.g., if it happened that both x-points were at dnAz and upAz,
  # then the flux between them would be simply bxFlux
  # However, if the x-points are elsewhere, we have to add the flux
  # between iFluxLeft and the x-point

  # Integrating B along the path from the lower x-point (along x) 
  #  to iFluxLeft, then straight up (in y) to the upper layer,
  # and over to the upper x-point, we get
  #  flux = A_z(upper x-point) - A_z(lower x-point) + bxFluxLeftXO
  # The lower x-point is at minimum A_z, the upper x-point at maximum:
  byFlux = upAz[:,:-1].max(axis=1) - dnAz[:,:-1].min(axis=1)

  #(omegac, rhoc) = getUnitScales(simName)
  #dnXpt = dnAz[:,:-1].argmin(axis=1)
  #dnXpt *= (upperBounds[0] - lowerBounds[0])/float(numPhysCells[0])
  #dnXpt += lowerBounds[0]
  #dnXpt /= rhoc
  #pylab.figure()
  #pylab.plot(byHistTimes*omegac, dnXpt)
  #plotPlus.setGoodSciLimits()
  #pylab.show()

  flux = - byFlux - bxFluxFn(byHistTimes)
  fluxFn = scipy.interpolate.interp1d(byHistTimes, flux, kind = "linear",
    bounds_error = False, fill_value = flux[-1])

  return fluxFn

def addFluxReconFrac(sDict): #{
  fluxFn = getUnreconnectedFluxVsTime(sDict["simName"])
  dt = sDict["DT"]
  nsteps = sDict["numSteps"]
  tmax = dt*nsteps
  fluxReconFrac = 1. - fluxFn(tmax)/fluxFn(0.)
  sDict["reconnectedFluxFrac"] = fluxReconFrac
  if 0: #{
    #print "frac reconnected flux = %g" % fluxReconFrac
    import pylab
    import sys
    ts = numpy.arange(0, tmax, dt)
    pylab.plot(ts , fluxFn(ts), '-')
    pylab.show()
    sys.exit()
  #}
  return fluxReconFrac
#}

def fit2reconRate(times, fluxes): #{
  """
  Given the curve fluxes vs. times (both 1D arrays), fits the curve to
  3 connected line segments, with the last being flat, and
  the first pinned to the point in (times, fluxes) when the flux is
  95% of its initial value.
  That leaves 4 variables: t_2, f_2, t_3, f_3 the coordinates
  of the boundaries between segments.

  Returns (rate1, rate2, [(t_1,f_1), (t_2,f_2), (t_3,f_3), (t_4, f_4)], fn)
  where rate1 and rate2 are the slopes of the 1st and 2nd segments,
  and the list is the segment endpoints: f_3=f_4=0,
  and fn is a function of t returning the fitted value.
  """
  t0 = times[0]
  f0 = fluxes[0]

  i1 = numpy.where(fluxes < 0.95*f0)[0][0]
  # 
  sf = arrayPlus.smooth1d(fluxes, max(1, i1/4), axis=0, endMethod="3/4,1/2,-1/4")
  i1 = numpy.where(sf < 0.95*f0)[0][0]
  t1 = times[i1]
  f1 = sf[i1]
  # we fit data only beyond i1
  
  t4 = times[-1]

  n = len(times)
  numSmooths = n/8
  sf = arrayPlus.smooth1d(fluxes, numSmooths, axis=0, endMethod="3/4,1/2,-1/4")
  sfmin = max(0, sf.min())
  # Guess values for t2, f2, t3, f3

  f3 = sfmin
  f4 = f3
  i3 = numpy.where(sf < 0.05*(f0-sfmin) + sfmin)[0][0]
  t3 = times[i3]

  t2 = 0.5*(t1+t3)
  f2 = 0.5*f1
  def pos(x): #{
    if (x>0):
      return x
    else:
      return 0
  #}
  def fitFn(t, t2, f2, t3, f3): #{
    res = 0 * t + f3
    d = pos(t1-t2+0.01*t4) + pos(t2-t3+0.01*t4) + pos(t3-2*t4)
    if (d > 0):
      res += (10+d)*f0
    else:
      if isinstance(t, numpy.ndarray) and t.size > 1:
        res[t<t3] = f3 + (t3-t[t<t3])/(t3-t2) * (f2 - f3)
        res[t<t2] = f1 - (t[t<t2]-t1)/(t2-t1) * (f1 - f2)
      else:
        if (t < t2):
          res = f1 - (t-t1)/(t2-t1) * (f1 - f2)
        elif (t < t3):
          res = f3 + (t3-t)/(t3-t2) * (f2 - f3)
        else:
          res = 0*t + f3
    return res
  #}
  def fitFn2(t, t2, f2): #{
    return fitFn(t, t2, f2, t3, f3)
  #}
  rtimes = times[i1:]
  rfluxes = fluxes[i1:]
  #print (t0,f0), (t1,f1), (t2,f2), (t3,f3), (t4,f4)
  # optimize t2, f2 only
  popt2 = (t2, f2)
  popt2 = scipy.optimize.curve_fit(fitFn2, rtimes, rfluxes, p0=popt2)[0]
  (t2, f2) = popt2
  # optimize over all
  popt = (t2, f2, t3, f3)
  popt = scipy.optimize.curve_fit(fitFn, rtimes, rfluxes, p0=popt)[0]
  (t2, f2, t3, f3) = popt
  # optimize t2, f2 only
  popt2 = (t2, f2)
  popt2 = scipy.optimize.curve_fit(fitFn2, rtimes, rfluxes, p0=popt2)[0]
  (t2, f2) = popt2
  # optimize over all
  popt = (t2, f2, t3, f3)
  popt = scipy.optimize.curve_fit(fitFn, rtimes, rfluxes, p0=popt)[0]
  (t2, f2, t3, f3) = popt
  rate1 = -(f2-f1)/(t2-t1)
  rate2 = -(f3-f2)/(t3-t2)
  if 0: #{
    import pylab
    import plotPlus
    pylab.plot(times, sf, '-', linewidth=3, alpha=0.4)
    pylab.plot(times, fluxes, ':', mfc='none')
    pylab.plot(rtimes, fitFn(rtimes, *popt), '-')
    pylab.title("rate1 = %g, rate2 = %g" % (rate1, rate2))
    plotPlus.setGoodSciLimits()
    pylab.show()
  #}
  def finalFn(t): #{
    return fitFn(t, *popt)
  #}
  f4 = f3
  return (rate1, rate2, numpy.array(
    [(t0,f0), (t1,f1), (t2,f2), (t3,f3), (t4, f4)]), finalFn)
  
#}

def addDistCharacteristicsKernel(sDict, 
  h, times, gammaBinEdges, timeSteps,
  histSliceSum = (), gammaM1 = False, estimateFinal = True,
  cachedFitData = None): #{
  """
  h is the history (which is a sum of histNames)
  times[i] is the corresponding physical times (in s) of h[i]
  timeSteps[i] are the indices first dim of the history that were 
    used to make h

  sDict needs: 
    omegac, tReconEnd, tReconEndPlusCrossTime

  estimateFinal = True takes avgs/medians from tReconEnd->tReconEndPlusCrossTime
                = False just takes avgs/medians of everything

  histSliceSum gives slices for GXYDist history, for x and y
  (but not for time, gamma, or ptcls/macronumber)
  """

  subtractLowEnergyMaxwellian = True
  fitCurve = (gammaM1 == False)
  findLongStraight = False

  omegac = sDict["omegac"]
  endOfReconTime = sDict["tReconEnd"]
  endOfReconTimePlus = sDict["tReconEndPlusCrossTime"]
  
  gs = 0.5*(gammaBinEdges[1:] + gammaBinEdges[:-1])
  m1 = 0.
  if gammaM1:
    gs -= 1.
    m1 = 1.
  dgs = gammaBinEdges[1:] - gammaBinEdges[:-1]
  gBinRatios = gammaBinEdges[1:]/gammaBinEdges[:-1]
  if (abs(1. - gBinRatios/gBinRatios[0]) > 1e-4).any(): 
    if not suppressOutput:
      print "Warning: bins do not have logarithmic widths, so median" + (
        " will probably yield a bad answer; should rebin before continuing")
  useSameGSmallest = False
  gSmallest = 0.01 if gammaM1 else 1.5
  # don't even consider stuff with g < 2.
  gsAll = gs
  dgsAll = dgs
  gBinRatiosAll = gBinRatios

  if fitCurve: #{
    fitPowers = []
    fitb1s = []
    fitb2s = []
    fitg1peaks = []
    fitg2peaks = []
    fitg3peaks = []
    fitFracInMaxLowE = []
    fitThetaLowE1 = []
    fitThetaLowE2 = []
  #}
  powers = []
  gAvgs= []
  g0peaks = []
  g1peaks = []
  g2peaks = []
  g3peaks = []
  gMins = []
  gMaxs = []
  modes = []
  longestPowers = []
  medians = []
  medianBetws = [] # median between min/max
  thLowE = []
  thLowEFail = []
  fracInMaxLowE = []
  energyFracInMaxLowE = []
  gAfterLowEMaxwellian = []
  weights = []
  macros = []
  for ti, t in enumerate(times): #{
    t = times[ti]
    dNAll = h[ti,:,0]
    dNmacro = h[ti,:,1]
    weights.append(dNAll.sum())
    macros.append(h[ti,:,1].sum())
    dNdgAll = dNAll/dgsAll
    if 1: #{ calculate low-energy maxwellian
      thbe = sDict["thetabe"]
      (dNdgMaxwell, amp, thFit, fracInMax, energyFracInMax, gMinInd, fail
        ) = subtractMaxwell(gammaBinEdges, dNdgAll, thbe)
      thLowE.append(thFit)
      thLowEFail.append(fail)
      fracInMaxLowE.append(fracInMax)
      energyFracInMaxLowE.append(energyFracInMax)
      gAfterLowEMaxwellian.append(gammaBinEdges[gMinInd])
      if not useSameGSmallest:
        gSmallest = gammaBinEdges[gMinInd]
    #}
    gsg2 = (gsAll>=gSmallest)
    gs = gsAll[gsg2]
    dgs = dgsAll[gsg2]
    gBinRatios = gBinRatiosAll[gsg2]
    dN = dNAll[gsg2]
    dNdg = dNdgAll[gsg2]
    if fitCurve: #{ calculate value from fitted curve
      thbe = sDict["thetabe"]
      startWithPrevFit = None
      if ti > 0:
        startWithPrevFit = popt
      if 0: #ti > 0 and (ti % 10 > 0):
        nz = (dNdgAll > 0.)
        lndNdg = numpy.log(dNdgAll[nz])
        lndNdgOld = numpy.log(fitFn(gsAll[nz]))
        #print lndNdg
        #print lndNdgOld
        diff = lndNdg - lndNdgOld
        avgDiff = abs(diff).sum()/diff.size 
        if avgDiff < numpy.log(1. + 0.2):
          startWithPrevFit = popt
      cdf = None
      if cachedFitData is not None: #{
        if t == 0.:
          cond = (cachedFitData[:,1] == t)
        else:
          cond = (abs(cachedFitData[:,1]/t - 1.) < 1e-6)
        iLines = numpy.where(cond)[0]
        if len(iLines) == 1.:
          cdf = cachedFitData[iLines[0],2:]
      #}
      (popt, pcov, fitFn, fitLowEnergyFn, fitPowerLawFn) = fit3(
        gammaBinEdges, dNdgAll, thbe, dNmacro,
        startWithPrevFit = startWithPrevFit,
        useCachedData = cdf)
      theta1 = popt[-2]
      theta2 = theta1 + popt[-1]
      dNdgPowLaw = fitPowerLawFn(gsAll)
      dNdgMaxwell = fitLowEnergyFn(gsAll)
      gdNdgPowLaw = gsAll * dNdgPowLaw
      ggdNdgPowLaw = gsAll * gdNdgPowLaw
      gggdNdgPowLaw = gsAll * ggdNdgPowLaw
      fitPowers.append(popt[1])
      fitb1s.append(popt[3])
      fitb2s.append(popt[4])
      gi = gdNdgPowLaw.argmax()
      fitg1peaks.append(gsAll[gi])
      gi = ggdNdgPowLaw.argmax()
      fitg2peaks.append(gsAll[gi])
      gi = gggdNdgPowLaw.argmax()
      fitg3peaks.append(gsAll[gi])
      nLo = (dNdgMaxwell * dgsAll).sum()
      nHi = (dNdgPowLaw * dgsAll).sum()
      fitFracInMaxLowE.append(nLo/(nLo+nHi))
      fitThetaLowE1.append(theta1)
      fitThetaLowE2.append(theta2)
    #}
    if findLongStraight: #{
      (nzLen, nzi1, nzi2, nzge1, nzge2) = findLongestConsecutiveTrue(
        dNdg > 0, numpy.log10(gammaBinEdges))
      if nzLen > 1: #{
        nzdNdg = dNdg[nzi1:nzi2]
        nzgs = gs[nzi1:nzi2]
        gm1Min = 0.5 + m1
        nzdNdg = nzdNdg[nzgs >= gm1Min]
        nzgs = nzgs[nzgs >= gm1Min]
        #print ti, t, omegac*t
        # if gm1 and dNdg is stored in equally-spaced log g bins, then
        # we don't have very many bins for gm1 << 1, and so we can't
        # smooth many times (not more than 4 times if including gm1 << 1).
        numSmooths = 8 if gammaM1 else 32
        for ns in [numSmooths]: #[0,2,4,16,64,256]:
          (alphaMode, loggLen, gStart) = findLongestPowerLawIndex(
            nzgs, nzdNdg, numSmooths = ns, indexSlop = 0.2)
          alphaMode *= -1.
          if 0:
            print "ns = %i, Longest alpha-stretch a = %.2f, g_2/g_1 = %.3g, g_1 = %.3g" % (
              ns, alphaMode, 10.**loggLen, gStart)
      else: #}{
        alphaMode = 0.
      #}
    #}
    gdNdg = gs * dNdg
    ggdNdg = gs * gdNdg
    gggdNdg = gs * ggdNdg
    if 0: #{
      import pylab
      import sys
      thbe = sDict["thetabe"]
      (dNdgMaxwell, amp, thFit, fracInMaxLoE, energyFracInMaxLoE, 
        gMinInd, fail) = subtractMaxwell(gammaBinEdges, dNdgAll, thbe)
      dNdgSub = abs(dNdgAll - dNdgMaxwell)
      gdNdgSub = gsAll * dNdgSub
      pylab.loglog(gsAll, dNdgAll, '.', label="dN/dg", alpha=0.5)
      pylab.loglog(gsAll, dNdgSub, ':', label="dif")
      pylab.loglog(gsAll, dNdgMaxwell, '-', label="maxwell")
      pylab.legend(loc='best')
      try:
        pylab.loglog([gs[gMinInd]], [dNdg[gMinInd]], 'o', mfc='none')
      except:
        pass
      #pylab.ylim(dNdg.min(), dNdg.max())
      pylab.ylim(ymin=dNdg.max()*1e-6)
      pylab.show()
      sys.exit()
    #}
    if 0: #{
      import pylab
      import sys
      pylab.figure()
      pc = pylab.loglog
      pc(gs, ggdNdg, label = '0')
      pc(gs, arrayPlus.smooth1d(ggdNdg,1), label = '1')
      pc(gs, arrayPlus.smooth1d(ggdNdg,4), label = '4')
      pc(gs, arrayPlus.smooth1d(ggdNdg,16), label = '16')
      pc(gs, arrayPlus.smooth1d(ggdNdg,64), label = '64')
      pc(gs, arrayPlus.smooth1d(ggdNdg,256), label = '256')
      pc(gs, arrayPlus.smooth1d(ggdNdg,1024), label = '1024')
      pc(gs, arrayPlus.smooth1d(ggdNdg,4096), label = '4096')
      pylab.legend()
      pylab.show()
    #}
    # smoothing
    ggdNdg = arrayPlus.smooth1d(ggdNdg, 256)
    N = dN.sum()
    gAvg = (gs * dN).sum() / N if N > 0 else 1.
    # find median slope of dN/dg vs. ln g
    # N.B. dNdg may be zero in some places, so lndNdg is -inf
    # smooth
    smdNdg = arrayPlus.smooth1d(dNdg, 0)
    if (smdNdg > 0).sum() > 4: #{
      smdNdg[smdNdg <= 0.] = 1e-210
      lndNdg = numpy.log(smdNdg)
      gSlopeBinCtrs = 0.5*(gs[1:]+gs[:-1])
      loglogSlope = -numpy.diff(lndNdg) / (gs[1:]-gs[:-1]) * gSlopeBinCtrs
      loglogSlope = loglogSlope[smdNdg[1:]*smdNdg[:-1] > 0]
      gSlopeBinCtrs = gSlopeBinCtrs[smdNdg[1:]*smdNdg[:-1] > 0]
      medianSlope = numpy.median(loglogSlope)
      # find mode slope
      slopeBinEdges = numpy.arange(numpy.floor(loglogSlope.min())-0.05,
        numpy.ceil(loglogSlope.max())+0.05+0.01, 0.1)
      slopeBins = 0.5*(slopeBinEdges[1:] + slopeBinEdges[:-1])
      slopeBinWeights = (gs[1:]/gs[:-1])[smdNdg[1:]*smdNdg[:-1] > 0]
      (slopeCount, slopeBinEdges) = numpy.histogram(loglogSlope,
        bins = slopeBinEdges, weights = slopeBinWeights)
      modeSlope = slopeBins[slopeCount.argmax()]
      #print "median slope = %g , mode slope = %g" % (medianSlope, modeSlope)
      if 0:
        import pylab
        pylab.figure()
        pylab.plot(slopeBins, slopeCount)
        pylab.show()

      if 0:
        import pylab
        pylab.figure()
        pylab.loglog(gs, dNdg, label = "0")
        pylab.loglog(gs, gdNdg, 'o', label = "1")
        pylab.loglog(gs, ggdNdg, label = "2")
        pylab.loglog(gs, gggdNdg, label = "3")
        print gAvg, gdNdg.min(), ggdNdg.max()
        pylab.loglog([gAvg, gAvg],[dNdg.max()/10., ggdNdg.max()], label = "gAvg")
        pylab.legend(loc='best')
        pylab.show()
        pylab.close()
      g0peak = gs[dNdg.argmax()]
      g1peak = gs[gdNdg.argmax()]
      g2peak = gs[ggdNdg.argmax()]
      #print g2peak
      g3peak = gs[gggdNdg.argmax()]
      gpeaks = [g0peak, g1peak, g2peak, g3peak]
      gpow = min(2, max(0, int(numpy.floor(medianSlope))))
      alwaysFitBetw1and2 = True
      if gpow != 1: #{
        wmsg = "Warning: power-law with index between %i and %i is " % (
          gpow, gpow+1)
        wmsg += "more prevalent."
        if alwaysFitBetw1and2:
           wmsg += "  However, we're going to fit for index between 1 and 2."
           gpow = 1
        if not suppressOutput:
          print "For history %i, time %g, " % (ti, t)
          print wmsg
      #}
      gMin = gpeaks[gpow]
      gMax = gpeaks[gpow+1]
      if gMin < gMax:
        gRatio = gMax/gMin
        gMinUp = gMin * gRatio**0.1
        gMaxDn = gMax * gRatio**(-0.1)
      else:
        gMinUp, gMaxDn = (gMax, gMin)
      inRange = numpy.logical_and(gs > gMinUp, gs < gMaxDn)
      inRange2 = numpy.logical_and(gSlopeBinCtrs > gMinUp, gSlopeBinCtrs < gMaxDn)
      if inRange.any(): 
        (a, C) = mathPlus.findPowerLaw(gs[inRange], dNdg[inRange])
        a *= -1.
      else:
        a = 1.
      if inRange2.any():
        medianBetw = numpy.median(loglogSlope[inRange2])
      else:
        medianBetw = 1.
    else: #}{
      g0peak = gSmallest
      g1peak = gSmallest
      g2peak = gSmallest
      g3peak = gSmallest
      gMin = gSmallest
      gMax = gSmallest
      modeSlope = 0.
      medianSlope = 0.
      medianBetw = 0.
      a = 0.
    #}
    g0peaks.append(g0peak)
    g1peaks.append(g1peak)
    g2peaks.append(g2peak)
    g3peaks.append(g3peak)
    gMins.append(gMin)
    gMaxs.append(gMax)
    powers.append(a)
    gAvgs.append(gAvg)
    modes.append(modeSlope)
    medians.append(medianSlope)
    medianBetws.append(medianBetw)
    if findLongStraight:
      longestPowers.append(alphaMode)
    else:
      longestPowers.append(1.)
  #}

  if fitCurve:
    fitPowers = numpy.array(fitPowers)
    fitb1s = numpy.array(fitb1s)
    fitb2s = numpy.array(fitb2s)
    fitg1peaks = numpy.array(fitg1peaks)
    fitg2peaks = numpy.array(fitg2peaks)
    fitg3peaks = numpy.array(fitg3peaks)
    fitFracInMaxLowE = numpy.array(fitFracInMaxLowE)
    fitThetaLowE1 = numpy.array(fitThetaLowE1)
    fitThetaLowE2 = numpy.array(fitThetaLowE2)

  powers = numpy.array(powers)
  g0peaks = numpy.array(g0peaks)
  g1peaks = numpy.array(g1peaks)
  g2peaks = numpy.array(g2peaks)
  g3peaks = numpy.array(g3peaks)
  gMins = numpy.array(gMins)
  gMaxs = numpy.array(gMaxs)
  gAvgs = numpy.array(gAvgs)
  modes = numpy.array(modes)
  longestPowers = numpy.array(longestPowers)
  medians = numpy.array(medians)
  medianBetws = numpy.array(medianBetws)
  thLowE = numpy.array(thLowE)
  thLowEFail = numpy.array(thLowEFail)
  fracInMaxLowE = numpy.array(fracInMaxLowE)
  energyFracInMaxLowE = numpy.array(energyFracInMaxLowE)
  gAfterLowEMaxwellian = numpy.array(gAfterLowEMaxwellian)
  weights = numpy.array(weights)
  macros = numpy.array(macros)

  stats = None
  if 1: #{ estimate "final values"
    if estimateFinal:
      iEndOfRecon = numpy.where(times >= endOfReconTime)[0][0]
      iEndOfRecon2 = numpy.where(times <= endOfReconTimePlus)[0][-1]
    else:
      iEndOfRecon = 0 
      iEndOfRecon2 = len(times)
    i1 = iEndOfRecon
    i2 = iEndOfRecon2
    graph=0
    def finalVal(ra): #{
      # better not take everything at the end, but some regular amount
      endRa = ra[iEndOfRecon:iEndOfRecon2]
      xs = numpy.arange(len(endRa))
      if 0 and estimateFinal:
        # allow slope
        (slope, endOfReconVal) = numpy.polyfit(xs, endRa, 1)
        endRaFit = slope*xs + endOfReconVal
      else:
        # just take average (or median)
        endOfReconVal = numpy.median(endRa)
        endRaFit = endOfReconVal
      diff = endRa - endRaFit
      if i2-i1 > 1:
        stdDev = (endRa - endRaFit).std(ddof=1)
      else:
        stdDev = 0.
      #maxDev = abs(endRa-endRaFit).max()
      #avg = endRa.mean()
      err = stdDev
      if graph: #{
        import pylab
        import sys
        pylab.plot(omegac*times[i1:i2], ra[i1:i2],'o')
        print "final val = %g, std = %g" % (endOfReconVal, err)
        pylab.show()
        sys.exit()
      #}
      return [endOfReconVal, err]
    #}
    stats = finalVal(g1peaks) + finalVal(g2peaks) + finalVal(powers) 
    stats += finalVal(medianBetws) 
    stats += finalVal(longestPowers)

    #estimate growth of theta_lowEnergyMaxwellian  after reconnection
    # assuming exponential growth
    if 1: #{
      thEnd = thLowE[i1:]
      timesEnd = times[i1:]
      if len(timesEnd) > 1:
        (slope, intercept) = numpy.polyfit(timesEnd, numpy.log(thEnd), 1)
        thGrowth = (slope, numpy.exp(intercept))
      else:
        thGrowth = (thEnd[0], 0.)
    #}
    weightStats = finalVal(weights)
    macroStats = finalVal(macros)
  #}

  if 1: #{ estimate values after saturation of g2
    g2ra = fitg2peaks if fitCurve else g2peaks
    try: #{
      (is1, isg2satp) = saturationPoint(g2ra, 0.9, logScale = False)
    except:
      is1 = iEndOfRecon
    is2 = min(is1 + iEndOfRecon2 - iEndOfRecon, len(times)-1)
    #print "is1, is2 = %i, %i" % (is1, is2)
    graph=0
    def satVal(ra): #{
      # better not take everything at the end, but some regular amount
      sRa = ra[is1:is2]
      sVal = numpy.median(sRa)
      sRaFit = sVal
      diff = sRa - sRaFit
      if i2-i1 > 1:
        stdDev = (sRa - sRaFit).std(ddof=1)
      else:
        stdDev = 0.
      #maxDev = abs(endRa-endRaFit).max()
      #avg = endRa.mean()
      err = stdDev
      if graph: #{
        import pylab
        import sys
        pylab.plot(omegac*times[i1:i2], ra[i1:i2],'o')
        print "final val = %g, std = %g" % (endOfReconVal, err)
        pylab.show()
        sys.exit()
      #}
      return [sVal, err]
    #}
  #}

  res = {
    "ptclWeight":weightStats[0],
    "ptclWeightStd":weightStats[1],
    "numMacros":macroStats[0],
    "numMacrosStd":macroStats[1],
    "powers":powers,
    "g0peaks":g0peaks,
    "g1peaks":g1peaks,
    "g2peaks":g2peaks,
    "g3peaks":g3peaks,
    "gMins":gMins,
    "gMaxs":gMaxs,
    "gAvgs":gAvgs,
    "modes":modes,
    "longestPowers":longestPowers,
    "medians":medians,
    "medianBetws":medianBetws,
    "g1f":stats[0:2],
    "g2f":stats[2:4],
    "powersf":stats[4:6],
    "medianBetwsf":stats[6:8],
    "longestPowersf":stats[8:10],
    "distCharTimes":times,
    "distCharEndOfReconTime":endOfReconTime,
    "distCharEndOfReconTimePlus":endOfReconTimePlus,
    "distCharEndOfReconTimeStep":iEndOfRecon,
    "distCharEndOfReconTimePlusStep":iEndOfRecon2,
    "thetaLowE":thLowE,
    "thetaLowEFail":thLowEFail,
    "fracInMaxLowE":fracInMaxLowE,
    "energyFracInMaxLowE":energyFracInMaxLowE,
    "gAfterMaxLowE":gAfterLowEMaxwellian,
    # theta_{low energy Maxwellian} = thGrowth[1] * exp(thGrowth[0] * t)
    "thetaLowEgrowth":thGrowth,
  }
  if fitCurve: #{
    fitRes = {
      "distCharg2SaturationTime":times[is1],
      "distCharg2SaturationTimePlus":times[is2],
      "distCharg2SaturationTimeStep":is1,
      "distCharg2SaturationTimePlusStep":is2,
      "fitPowers":fitPowers,
      "fitb1s":fitb1s,
      "fitb2s":fitb2s,
      "fitg1peaks":fitg1peaks,
      "fitg2peaks":fitg2peaks,
      "fitg3peaks":fitg3peaks,
      "fitFracInMaxLowE":fitFracInMaxLowE,
      "fitThetaLowE1":fitThetaLowE1,
      "fitThetaLowE2":fitThetaLowE2,
      "fitPowersf":finalVal(fitPowers)[0],
      "fitPowersfstd":finalVal(fitPowers)[1],
      "fitPowerse":satVal(fitPowers)[0],
      "fitPowerseStddev":satVal(fitPowers)[1],
      "fitb1sf":finalVal(fitb1s)[0],
      "fitb2sf":finalVal(fitb2s)[0],
      "fitg1peaksf":finalVal(fitg1peaks)[0],
      "fitg2peaksf":finalVal(fitg2peaks)[0],
      "fitg3peaksf":finalVal(fitg3peaks)[0],
      "fitFracInMaxLowEf":finalVal(fitFracInMaxLowE)[0],
      "fitThetaLowE1f":finalVal(fitThetaLowE1)[0],
      "fitThetaLowE2f":finalVal(fitThetaLowE2)[0],
    }
    for (k,v) in fitRes.iteritems():
      res[k] = v
  #}

  for (k,v) in res.iteritems():
    kp = k + "gm1" if gammaM1 else k
    sDict[kp] = v
  return res
#}

def writeFits(wfFile, sDict, h, times, gEdges): #{
  omegac = sDict["omegac"]
  endOfReconTime = sDict["tReconEnd"]
  endOfReconTimePlus = sDict["tReconEndPlusCrossTime"]
  
  gs = 0.5*(gEdges[1:] + gEdges[:-1])
  dgs = gEdges[1:] - gEdges[:-1]
  s = "step, time (s), lnC0, a0, a20, b10, b20, c10, d0, lnMC0, th10, dth0, gMinFit, gMaxFit\n"
  havePopt = False
  for ti, t in enumerate(times): #{
    t = times[ti]
    dN = h[ti,:,0]
    (igMinFit, igMaxFit) = numpy.where(dN > 0.)[0][[0,-1]]
    dNdg = dN/dgs
    dNmacro = h[ti,:,1]
    thbe = sDict["thetabe"]
    startWithPrevFit = popt if havePopt else None
    (popt, pcov, fitFn, fitLowEnergyFn, fitPowerLawFn) = fit3(
      gEdges, dNdg, thbe, dNmacro,
      startWithPrevFit = startWithPrevFit)
    havePopt = True
    s += ",".join([str(ti), repr(t)] + [repr(p) for p in popt]
      + [repr(gs[igMinFit]), repr(gs[igMaxFit])]) + "\n"
  #}
  with open(wfFile, 'w') as f:
    f.write(s)
#}
  
def combBins(h, gammaBinEdges, binsPerDecade,
  arbitrariness = {"binsPerDecade":1.}): #{
  curLnRatio = numpy.log(gammaBinEdges[-1]/gammaBinEdges[0])/h.shape[1]
  desLnRatio = numpy.log(10.)/(binsPerDecade*arbitrariness["binsPerDecade"])
  factor = int(numpy.round(desLnRatio/curLnRatio))
  (hC, gammaBinEdgesC) = arrayPlus.compactHistogram(h, gammaBinEdges,
    factor, binDir = 1)
  print "combining groups of %i bins" % factor
  if 1: #{
    # This approach is a bit doubtful...
    # It might work if the noise in each bin is independent of the next,
    # but that is not the case, at least for sufficiently nearby bins.
    hSmooth = arrayPlus.smooth1d(h, 2*factor, axis = 1)
    hDevSqr = (h - hSmooth)**2 
    # Now we've estimated noise in the original bins, but
    # we really want to estimate the noise in the final bins;
    # we do that simply assuming that the noise is reduced by
    # 1/sqrt(numOrigBinsPerNewBin)
    hDevSqrC = arrayPlus.compactHistogram(hDevSqr, gammaBinEdges,
      factor, binDir = 1)[0]
    hStdC = numpy.sqrt(hDevSqrC)
    if 0: #{
      import pylab
      print h is hSmooth, (h==hSmooth).all()
      g = 0.5*(gammaBinEdges[1:] + gammaBinEdges[:-1])
      pylab.plot(g, h[0,...], 'o')
      pylab.plot(g,hSmooth[0,...], '-')
      pylab.show()
      raise RuntimeError, "Quitting after debugging plot"
    #}
  #}
  if 0: #{
    # This approach is a bit doubtful...
    # It might work if the noise in each bin is independent of the next,
    # but that is not the case, at least for sufficiently nearby bins.
    numPerBin = factor * numpy.ones((len(gammaBinEdgesC)-1,),
      dtype = numpy.int64)
    numPerBin[-1] -= (numPerBin.sum() - len(gammaBinEdges) + 1)
    if numPerBin.sum() != len(gammaBinEdges)-1:
      raise RuntimeError, "Bug! Fix this."
    gC = 0.5*(gammaBinEdgesC[1:] + gammaBinEdgesC[:-1])
    dhCdgC = hC/numpy.diff(gammaBinEdgesC)
    dhCdgCinterp = scipy.interpolate.interp1d(gC, dhCdgC, 
      kind = "linear", axis=1, 
      bounds_error = False, fill_value = dhCdgC[:,-1,...],
      #assume_sorted = True
      )
    g = 0.5*(gammaBinEdges[1:] + gammaBinEdges[:-1])
    hSmooth = dhCdgCinterp(g) * numpy.diff(gammaBinEdges)
    if 0: #{
      import pylab
      pylab.plot(g, h[0,...], 'o')
      pylab.plot(g,hSmooth[0,...], '-')
      pylab.plot(g, arrayPlus.smooth1d(h[0,...], factor), '-r')
      pylab.show()
      raise RuntimeError, "Quitting after debugging plot"
    #}
    hDevSqr = (h - hSmooth)**2 
    # Now we've estimated noise in the original bins, but
    # we really want to estimate the noise in the final bins;
    # we do that simply assuming that the noise is reduced by
    # 1/sqrt(numOrigBinsPerNewBin)
    hDevSqrC = arrayPlus.compactHistogram(hDevSqr, gammaBinEdges,
      factor, binDir = 1)[0]
    #print "h       =", h
    #print "hSmooth =", hSmooth
    #print "hDevSqrC =", hDevSqrC
    # extrapolation to low g is bad because fill_value is for high g
    hDevSqrC[:,0] = hDevSqrC[:,1]
    hStdC = numpy.sqrt(hDevSqrC)
    #hStdC = hDevSqrC.copy()
    #hStdC[numPerBin<=1] = 0.
    #numPerBin[numPerBin<=1] = 2.
    #hStdC /= (numPerBin-1.)
    #hStdC = arrayPlus.smooth1d(hDevSqrC, 1, axis = 1)
    #hStdC = numpy.sqrt(hStdC)
  #}
  return (hC, gammaBinEdgesC, hStdC)
#}

if 0: #{ test combBins
  numBins = 400
  binsPerDecade = 50.
  binEdges = 10**(numpy.arange(numBins+1)/binsPerDecade) 
  binCtrs = 0.5*(binEdges[1:] + binEdges[:-1])
  ys = 1. + numpy.sqrt(binCtrs)
  ys = numpy.reshape(ys, (1,) + ys.shape)
  sigmas = numpy.ones(ys.shape)
  sigmas *= 0.1
  sys = ys.copy()
  for i,y in enumerate(ys):
   sys[i] = numpy.random.normal(y, y*sigmas[i])

  cbinsPerDecade = 5
  (cys, cbinEdges, cSigmas) = combBins(sys, binEdges, cbinsPerDecade)
  combFactor = (binEdges.size-1.)/ (cbinEdges.size-1.)
  print sigmas
  print cSigmas/cys
  print cSigmas/cys * numpy.sqrt(combFactor)

  raise RuntimeError, "Quitting after debugging unit test."

#}

def betterDistData(h, gammaBinEdges, timeSteps, combineBins = True,
  lnNvarOverTime = 0.1, 
  arbitrariness = {"lnNvar":1., "binsPerDecade":1, "gMin":1., "gMax":1.}): #{
  """
  h[step, ig, 0] = int_gammaBinEdges[ig]^gammaBinEdges[ig+1] dN/dg dg
  h[...,1] = number of macroparticles
  """
  

  hOrig = h

  if lnNvarOverTime is not None: #{
    lnNvar = lnNvarOverTime * arbitrariness["lnNvar"]
    dN = h[...,0]
    g = 0.5*(gammaBinEdges[1:] + gammaBinEdges[:-1])
    gdN = g[numpy.newaxis,:]*dN
    N = dN.sum(axis=1)
    hAvg = numpy.zeros((len(timeSteps),) + h.shape[1:])
    hStd = hAvg.copy()
    (hC, gammaBinEdgesC) = combBins(h, gammaBinEdges, 5, 
      arbitrariness=arbitrariness)[:2]
    dgC = numpy.diff(gammaBinEdgesC)
    dNdgC = hC[...,0] / dgC
    gC = 0.5*(gammaBinEdgesC[1:] + gammaBinEdgesC[:-1])
    totalSteps = h.shape[0]
    print "totalSteps", totalSteps
    for tii, ti in enumerate(timeSteps): #{
      (gMin, gMax, igMin, igMax) = gMinMaxForTailFit(
        gammaBinEdges, dN[ti], arbitrariness = arbitrariness)
      igMinC = numpy.where(gammaBinEdgesC < gMin)[0][-1]
      igMaxC = numpy.where(gammaBinEdgesC > gMax)[0][0]
      lnh = numpy.log(hC[:,igMinC:igMaxC,0])
      udist = 0
      ldist = 0
      print ti, gMin, gMax, igMinC, igMaxC
      while (ti - ldist-1 >= 0) or (ti+udist+1 < totalSteps): #{
        if (ti - ldist - 1 >= 0) and (ti + udist + 1 < totalSteps):
          dif = 0.5*(lnh[ti-ldist-1]+lnh[ti+udist+1]) 
          ldistInc = 1
          udistInc = 1
        elif (ti - ldist - 1 >= 0):
          dif = lnh[ti-ldist-1]
          ldistInc = 1
          udistInc = 0
        else:
          dif = lnh[ti+udist+1]
          ldistInc = 0
          udistInc = 1
        dif -= lnh[ti]
        #sdifMeas = dif.sum() / len(dif)
        adifMeas = abs(dif).sum() / len(dif)
        #print ti, ldist+1, udist+1, adifMeas
        #print "...", dif
        if adifMeas <= lnNvar:
          ldist += ldistInc
          udist += udistInc
        else:
          break
      #}
      print "timestep", ti, "averaging from", ti-ldist, "to", ti+udist
      hAvg[tii] = h[ti-ldist:ti+udist+1].mean(axis=0)
      dist = min(ldist, udist)
      if dist > 0: 
        hStd[tii] = numpy.std(h[ti-dist:ti+dist+1], ddof=1, axis=0)
      else:
        hStd[tii] = 0.
      # We really want the expected std dev of hAvg
      hStd[tii] /= numpy.sqrt(1.+2*dist)
    #}
    h = hAvg
    hStd = hStd[...,0]
  else: #}{
    hStd = None
  #}

  if combineBins: #{
    cnumBins = 20
    (hC, gammaBinEdgesC, hStdC) = combBins(h, gammaBinEdges, cnumBins,
      arbitrariness=arbitrariness)

    if 1: #{
      hStdCrel = estimateDistError(hC[...,0], hC[...,1], gammaBinEdgesC)
      if 0: #{
        import pylab
        g = 0.5*(gammaBinEdges[1:] + gammaBinEdges[:-1])
        gC = 0.5*(gammaBinEdgesC[1:] + gammaBinEdgesC[:-1])
        dg = numpy.diff(gammaBinEdges)[numpy.newaxis,:]
        dgC = numpy.diff(gammaBinEdgesC)[numpy.newaxis,:]
        dNdg = h[:,:,0] / dg
        dNdgC = hC[:,:,0] / dgC
        dNdgErr = hStdCrel * dNdgC / dgC
        tii = 0
        ti = timeSteps[tii]
        plotCmd = pylab.loglog
        def pc(x, y, *args): #{
          p = 1.5
          plotCmd(x, y * x**p, *args)
        #}
        pc(g, dNdg[tii,:], '.r')
        pc(gC, dNdgC[tii,:], '-b')
        nstd = 3
        pc(gC, dNdgC[tii,:] - nstd*dNdgErr[tii,:], '--b')
        pc(gC, dNdgC[tii,:] + nstd*dNdgErr[tii,:], '--b')
        pylab.show()
        raise RuntimeError, "Quitting after diagnostic plot"
      #}
    #}
    if 0: #{
      gC = 0.5*(gammaBinEdgesC[1:] + gammaBinEdgesC[:-1])
      dgC = numpy.diff(gammaBinEdgesC)
      dNdgC = hC / dgC[numpy.newaxis,:,numpy.newaxis]
      dNdgSmoothC = arrayPlus.smooth1d(dNdgC, 4, axis = 1)
      hSmoothC = dNdgSmoothC * dgC[numpy.newaxis,:,numpy.newaxis]
      #varC = (hC-hSmoothC)**2
      #normVarC = varC / hSmoothC**2
      hStdC3 = numpy.sqrt(arrayPlus.smooth1d((hC-hSmoothC)**2, 5, axis=1))
      hStdC = hStdC3
      hStdC = hStdC[...,0]
      if hStd is not None: #{
        hStdC2 = numpy.sqrt(combBins(hStd**2, gammaBinEdges, cnumBins,
          arbitrariness=arbitrariness)[0])
        if 0: #{
          import pylab
          #print "hStdC  =", hStdC 
          #print "hStdC2 =", hStdC2
          #print "hStdC/hStdC2 =", hStdC/hStdC2
          dg = numpy.diff(gammaBinEdges)[numpy.newaxis,:]
          dgC = numpy.diff(gammaBinEdgesC)[numpy.newaxis,:]
          dNdg = h[:,:,0] / dg
          dNdgC = hC[:,:,0] / dgC
          dNdgErr = hStdC / dgC
          dNdgErr2 = hStdC2 / dgC
          print "median(hStdC)  =", numpy.median(hStdC[...,0]/hC[...,0])
          print "median(hStdC3) =", numpy.median(hStdC3[...,0]/hC[...,0])
          print "median(hStdC2) =", numpy.median(hStdC2[...,0]/hC[...,0])
          print "median(dNdgErr) norm  =", numpy.median(dNdgErr/dNdgC)
          print "median(dNdgErr2) norm =", numpy.median(dNdgErr2/dNdgC)
          tii = 0
          ti = timeSteps[tii]
          print g.shape, hOrig.shape
          print gC.shape, hC.shape
          plotCmd = pylab.loglog
          def pc(x, y, *args): #{
            p = 1.5
            plotCmd(x, y * x**p, *args)
          #}
          pc(g, dNdg[tii,:], '.r')
          pc(gC, dNdgC[tii,:], '-b')
          pc(gC, dNdgSmoothC[tii,:,0], '-g')
          nstd = 3
          pc(gC, dNdgC[tii,:] - nstd*dNdgErr[tii,:], '--b')
          pc(gC, dNdgC[tii,:] + nstd*dNdgErr[tii,:], '--b')
          pc(gC, dNdgC[tii,:] - nstd*dNdgErr2[tii,:], ':b')
          pc(gC, dNdgC[tii,:] + nstd*dNdgErr2[tii,:], ':b')
          pylab.show()
          raise RuntimeError, "Quitting after diagnostic plot"
        #}
      #}
    #}
    #raise RuntimeError, "Quitting"
      
    h = hC
    gammaBinEdges = gammaBinEdgesC
    hStdRel = hStdCrel
  #}

  return (h, gammaBinEdges, hStdRel)
  
#}

def getMultipleSimDists(simNames, histNames, timeSteps = None,
  histSliceSum = (), combineBins = True,
  avgOverTime = None, 
  arbitrariness = {"lnNvar":1., "binsPerDecade":1., "gMin":1., "gMax":1.,
  "relVar":1.}): #{
  """
  simNames is a list of simulations, 
    e.g., ["relRecon2p", "../m1-sig100/relRecon2p"]
  histNames is either:
    - a nested list of history names, with one list of history names
      per element of simNames (i.e., per simulation)
    - a single list of history names to be used for each simulation
  Each simulation history must have exactly the same size/shape.
  timeSteps = a list of steps (first index of the History) at which to
    find the distributions, or None in which case all timeSteps are used.
  histSliceSum = (complicated option...see the gammaDistInLayer options
    of graphPtclHist.py to see how to use)

  avgOverTime = lnNvar * arbitrariness["lnNvar"]
                where the distribution for time t will be averaged 
                over nearby (in time) distributions such that the
                average difference (between different times) in ln(dN/dg) 
                over the range [gMin, gMax] is less than lnNvar 

                suggested value for lnNvar = 0.1 ( ~ 10% variation in N)
  arbitrariness is used only if avgOverTime is

  Returns (h, times, gammaBinEdges) 
    where h is the sum of all the individual histories, 
    an array the same size as each history, except that the first 
    dimension has size len(timeSteps) if timeSteps is given, 
    and times[i] is the time (in seconds) of history record h[i].
  """
  sim0 = simNames[0]
  if not (isinstance(histNames[0], list) or isinstance(histNames[0], tuple)):
    histNames = [histNames] * len(simNames)
  #try:
  #  histNames[0][0]
  #except:
  #  histNames = [histNames] * len(simNames)
  hist0 = histNames[0][0]
  gammaBinEdges = vorpalUtil.getHistoryHdf5Attr(sim0, hist0, "gammaBinEdges")
  gMax = gammaBinEdges[-1]
  sh = list(vorpalUtil.getHistoryShape(sim0, hist0))
  sh[1] = gammaBinEdges.size
  if timeSteps is None:
    timeSteps = range(sh[0])
  origTimeSteps = numpy.array(timeSteps)
  if avgOverTime is not None:
    timeSteps = range(sh[0])
    if origTimeSteps.max() > timeSteps[-1]:
      msg = "Step %i is greater than maximum (%i)" % (origTimeSteps.max(),
        timeSteps[-1])
      raise ValueError, msg
  print "timeSteps", timeSteps
  # steps x gammaBins x 2 (2 for num particles, num macroparticles)
  numSims = len(simNames)
  numHists = len(histNames)
  numSteps = len(timeSteps)
  # don't graph last bin, which just contains everything over gammaMax
  h = numpy.zeros((numSteps, sh[1]-1,2), dtype=numpy.float64)
  for si, simName in enumerate(simNames): #{
    for hi, histName in enumerate(histNames[si]): #{
      hasSpilloverBin = True
      try:
        if len(histSliceSum) == 0:
          h1 = vorpalUtil.getHistory(simName, histName)[timeSteps]
        else:
          slices = [slice(None), slice(None)] + list(histSliceSum) + [slice(1)]
          h1 = vorpalUtil.getHistory(simName, histName, slices)[timeSteps]
          # GXY histories had bug leaving out bin for particles exceeding
          #   max gamma...add it here to avoid error
          if h1.shape[1] < gammaBinEdges.size:
            if not suppressOutput:
              print "Warning: %s doesn't have bin for gamma exceeding max" % histNames[hi]
            hasSpilloverBin = False
          # h1 may not have number of macro-particles (in case of constant weight ptcls)
          if h1.shape[-1] < 2: #{
            # assume constant weight
            posNames = ["".join(s) for s in itertools.product(
                ("dn","up"), ("bg", "drift"), ("Electrons", "Ions", "Positrons"))]
            speciesName = None
            for spName in posNames:
              if re.match(spName, histName):
                speciesName = spName
            if speciesName is None:
              msg = "Cannot determine which species yielded " + histName 
              raise ValueError, msg
            sprops = vorpalUtil.getSpeciesProperties(simName, speciesName)
            h1MacroNumber = h1 / sprops["macroNumber"]
            h1 = numpy.concatenate((h1, h1MacroNumber), axis=-1)
          #}
          # sum over histSliceSum
          while len(h1.shape) > 3:
            h1 = h1.sum(axis = -2)
        if hasSpilloverBin and len(h1) > 2 and h1.shape[-1]>1 and (h1[:,-1,1] > 0).any():
          if not suppressOutput:
            print "Warning:", simName, ":", histName, 
            print "had particles exceeding the maximum gamma-bin."
      except:
        print histName + ".shape =", sh
        raise
      if hasSpilloverBin:
        h += h1[:,:-1,:]
      else:
        h += h1[:,:,:]
    #}
  #}

  if 1: #{ eliminate zeros, because we like to take logs
    hMin = h[...,0].min()
    h[...,0][h[...,0]==0] = 0.1*hMin
  #}
  
  arbitrariness = "all"
  #arbitrariness = {"lnNvar":1., "binsPerDecade":1., "gMin":1., "gMax":1.}
  fit = True
  if fit and arbitrariness == "all": #{
    lnNvar = (1.,) if avgOverTime is None else (0.5, 1., 2.)
    res = []
    prmra = []
    #iterOver = ((2.,), (2.,), (1.2,), (0.1,), (0.5,))
    #iterOver = ((1.,), (1.,2.), (1.,1.25), (0.1,1.), (1.,1.5))
    arbNames = ("lnNvar", "binsPerDecade", "gMin",  "gMax",    "relVar")
    iterOver = (lnNvar, (1.,2.), (0.75,1.,1.25), (0.1,1.,10.), (0.5,1.,1.5))
    #iterOver = (lnNvar, (1.,2.), (0.75,1.,1.25), (0.001, 0.01,0.1,1.,10.), (0.5,1.,1.5))
    for arbs in itertools.product(*iterOver): #{
      #arb = {"lnNvar":arbs[0], "binsPerDecade":arbs[1], "gMin":arbs[2],
      #  "gMax":arbs[3], "relVar":arbs[4]}
      arb = dict(zip(arbNames, arbs))
      (h2, gammaBinEdges2, hStdRel2) = betterDistData(
        h, gammaBinEdges, origTimeSteps, combineBins = True,
        lnNvarOverTime = avgOverTime, 
        arbitrariness = arb)
      for tii, ti in enumerate(origTimeSteps):
        (alpha, gc1best, gc2best, fitFn, 
         fitGammaRange, fitStdDev, gCrits,
         igBest, fitFns, poptsA, pcovsA, rmsChis) = fit6(
          gammaBinEdges2, 
          h2[tii,...,0]/numpy.diff(gammaBinEdges2), 
          h2[tii,...,1],
          hStdRel2[tii,...], arbitrariness=arb)
        (gc1, gc2, gcb1, gcb2) = gCrits
        #res.append((ti, alpha, gc, oddsOfExp, chiSqr, chiSqrTail, arb))

        res.append((ti, alpha, gc1best, gc2best, fitGammaRange[0],
          fitGammaRange[1], igBest, gc1, gc2, gcb1, gcb2, arb))
        prmra.append(arbs)

    #}
    resra = []
    for row in res:
      resra.append(row[:-1])
    resra = numpy.array(resra)
    prmra = numpy.array(prmra)
    iAlpha = 1
    igc1 = 2
    igc2 = 3
    igLo = 4
    igHi = 5
    # weight: since alpha =  - (ln f_2 - ln f_1)/(ln g_2 - ln g_1)
    #  (f=dN/dg) then assuming similar delta ln f, the error in alpha
    #  is proportional to 1/(ln g_2 - ln g_1).
    #wAlpha = numpy.log(res[:,igHi]) - numpy.log(res[:,igLo])
    # 1 SD is 34.1% below/above mean
    qs = [50.-34.1, 50., 50.+34.1]
    alphaRange = numpy.percentile(resra[:,iAlpha], qs)
    gc1Range = numpy.percentile(resra[:,igc1], qs)
    gc2Range = numpy.percentile(resra[:,igc2], qs)
    
    excludeOutliers = True
    if excludeOutliers: #{ #exclude outliers
      def findWithinDevs(ra, col, valRange, devs = 3, log = False): #{
        """
        """
        rac = ra[:,col]
        if log: #{
          valRange = numpy.log(valRange)
          rac = numpy.log(rac)
        #}
        (lo, med, hi) = valRange
        yLo = med + devs * (lo-med)
        yHi = med + devs * (hi-med)
        yMid = (yLo+yHi)/2.
        dy = (yHi-yLo)/2.
        inDevs = ( abs(rac-yMid) <= dy )
        return inDevs
      #}
      def inDevsToChar(*inDevs): #{
        n = len(inDevs)
        s = []
        for i in range(len(inDevs[0])): #{
          ss = ""
          for j in range(n): #{
            ss += ("x", " ")[inDevs[j][i]]
          #}
          s.append(ss)
        #}
        return s
      #}
      inAlphaDevs = findWithinDevs(resra, iAlpha, alphaRange)
      inGc1Devs = findWithinDevs(resra, igc1, gc1Range, log=True)
      inGc2Devs = findWithinDevs(resra, igc2, gc2Range, log=True)
      # special case: exclude cases with gc2 <= gc2MinAccept
      if 0: #{
        #inGc2Devs = numpy.logical_and(inGc2Devs, 
        #  resra[:,igc2] > 1000.)
        gc2MinAccept = 5000.
        print 'Excluding results with gamma_c2 < %.2g' % gc2MinAccept
        inGc2Devs = (resra[:,igc2] > gc2MinAccept)
      #}
      inDevChars = inDevsToChar(inAlphaDevs, inGc1Devs, inGc2Devs)

      inDevs = numpy.logical_and(inGc1Devs, inGc2Devs)
      inDevs = numpy.logical_and(inAlphaDevs, inDevs)



      resra = resra[inDevs]
      prmra = prmra[inDevs]
      alphaRange = numpy.percentile(resra[:,iAlpha], qs)
      gc1Range = numpy.percentile(resra[:,igc1], qs)
      gc2Range = numpy.percentile(resra[:,igc2], qs)

    else: #}{
      inDevs = numpy.ones(resra.shape[:1], dtype=numpy.bool)
    #}

    if 1: #{ calculate importance of arbitrary parameters in 
      # explaining spread, by trying linear regression
      # Let x = [x_0, ..., x_{N-1}] be N arbitrary parameters,
      # and f(x) = the result \approx x^T q + f_0
      #  where q = [q_0, ..., q_{N-1}] and f_0 is a scalar
      # or f(x) \approx [1, x]^T [f_0, q] 
      # If the linear model were perfect, then there would be an f_0 and q
      #   s.t. f_i = f(x^i) = [1, x^i]^T [f_0, q]  for all choices of
      #   parameters x^i = [x^i_0, ..., x^i_{N-1}].
      # Writing X_{ij} = x^i_j, this is equivalent to
      #   f_i = X^T [f_0, q] = X^T q'
      # We find q' in a least-squares sense by q' = [X^T]^{-1} f.
      # However, we have to normalize our parameters, and sometimes take
      #   the log.  Actually, we've already chosen a normalization by
      #   choosing which parameters to explore.  And we're not looking
      #   for a precise fit, just order of magnitude importance.
      numArbParams = len(arbNames)
      normPrmRa = numpy.hstack((numpy.ones((prmra.shape[0],1)), prmra.copy()))
      # replace params with normalized values
      for pi in range(numArbParams): #{
        choices = iterOver[pi]
        col = normPrmRa[:,1+pi]
        # replace, but make sure replacements are unique
        colMax10 = max(1., 10.*abs(col).max())
        if len(choices) > 1: #{
          for iCh, ch in enumerate(choices): #{
            col[col==ch] = (iCh+1) * colMax10
          #}
          # now renormalize to integers
          col /= colMax10
        else: #}{
          col *= 0.
        #}
      #}
      print "gradients in parameter space", arbNames
      for fii, fi in enumerate((iAlpha, igc1, igc2)): #{
        grad = numpy.linalg.lstsq(
          normPrmRa, resra[:,fi], rcond=1e-4)[0]
        # toss away intercept
        grad = grad[1:]
        resName = ["alpha", "gc1", "gc2"][fii]
        opo = numpy.get_printoptions()
        numpy.set_printoptions(precision=2)
        print "gradient for", resName + ":", grad
        numpy.set_printoptions(**opo)
      #}
    #}


    if 1: #{
      ra = []
      print "a12 step lnNvar bins/Decade gMin gMax relVar:",
      print "alpha gc1  gc2  g_start g_stop"
      for ri, (s,a,g1,g2,gb,ge,ib,gc1,gc2,gcb1,gcb2,arb) in enumerate(res): #{
        ra.append((s,
          arb["lnNvar"], arb["binsPerDecade"], arb["gMin"], arb["gMax"],
          arb["relVar"],
          a, g1, g2, gb, ge))
        print (inDevChars[ri]
          ) + "%5i %5.2g %5.3g  %5.2g %5.2g %5.2g:  %5.3g %7.2g %7.2g %7.2g %7.2g" % ra[-1]
      #}
      ra = numpy.array(ra)
      print "alpha: %.3g -> %.3g, avg = %.3g, med = %.3g, sd = %.3g" % (
        ra[:,-5].min(), ra[:,-5].max(), ra[:,-5].mean(), numpy.median(ra[:,-5]),
        numpy.std(ra[:,-5], ddof=1))
      print "gc1: %.3g -> %.3g, avg = %.3g, med = %.3g, sd = %.3g" % (
        ra[:,-4].min(), ra[:,-4].max(), ra[:,-4].mean(), numpy.median(ra[:,-4]),
        numpy.std(ra[:,-4], ddof=1))
      print "gc2: %.3g -> %.3g, avg = %.3g, med = %.3g, sd = %.3g" % (
        ra[:,-3].min(), ra[:,-3].max(), ra[:,-3].mean(), numpy.median(ra[:,-3]),
        numpy.std(ra[:,-3], ddof=1))
      
      if excludeOutliers:
        print "Excluding outliers"
      print "alpha   = %.4g %-.4g %+.4g" % (
        alphaRange[1], alphaRange[0]-alphaRange[1], alphaRange[2]-alphaRange[1])
      print "gc1   = %.4g %-.4g %+.4g" % (
        gc1Range[1], gc1Range[0]-gc1Range[1], gc1Range[2]-gc1Range[1])
      print "gc2   = %.4g %-.4g %+.4g" % (
        gc2Range[1], gc2Range[0]-gc2Range[1], gc2Range[2]-gc2Range[1])
    #}
    arbitrariness = {"lnNvar":1., "binsPerDecade":1., "gMin":1., "gMax":1.}
  #}
  (h, gammaBinEdges, hStdRel) = betterDistData(
    h, gammaBinEdges, origTimeSteps, combineBins = True,
    lnNvarOverTime = avgOverTime, 
    arbitrariness = arbitrariness)

  timeSteps = origTimeSteps

  #raise RuntimeError, "stopping for debugging"

  try:
    stepsAndTimes = vorpalUtil.getHistory(sim0, "smallDistTimes")
    if stepsAndTimes.shape[0] != sh[0]:
      if not suppressOutput:
        print "Using largeDistTimes"
      stepsAndTimes = vorpalUtil.getHistory(sim0, "largeDistTimes")
      if stepsAndTimes.shape[0] != sh[0]:
        if not suppressOutput:
          print "can't find right dist times"
        stepsAndTimes = numpy.zeros((sh[0],2), dtype=numpy.float64)
    times = stepsAndTimes[:,0]
    times = times[timeSteps]
        
  except:
    print "Warning: DistTimes not found"
    raise ValueError, "DistTimes not found"
    times = numpy.array(timeSteps)*1.86805e-9


  #print h.shape
  #print gammaBinEdges.shape
  return (h, times, gammaBinEdges)
#}


def addDistCharacteristics(sDict, histNames, timeSteps = None,
  histSliceSum = (), gammaM1 = False, estimateFinal = True,
  writeFitsFile = None, cachedFitData = None): #{
  """
  if writeFits, just writes a file with best fit parameters
  """
  if writeFitsFile is not None and (gammaM1 == "both" or gammaM1 == True):
    raise ValueError, "cannot make fits for gammaM1"

  simNames = [sDict["simName"]]
  hist0 = histNames[0]
  simName0 = simNames[0]
  sameSim = (len(set(simNames)) == 1)
  gammaBinEdges = vorpalUtil.getHistoryHdf5Attr(simNames[0], histNames[0],
    "gammaBinEdges")
  gMax = gammaBinEdges[-1]
  sh = list(vorpalUtil.getHistoryShape(simName0, hist0))
  sh[1] = gammaBinEdges.size
  if timeSteps is None:
    timeSteps = range(sh[0])
  # steps x gammaBins x 2 (2 for num particles, num macroparticles)
  numHists = len(histNames)
  numSteps = len(timeSteps)
  # don't graph last bin, which just contains everything over gammaMax
  h = numpy.zeros((numSteps, sh[1]-1,2), dtype=numpy.float64)
  for hi in range(numHists):
    hasSpilloverBin = True
    try:
      if len(histSliceSum) == 0:
        h1 = vorpalUtil.getHistory(simNames[0], histNames[hi])[timeSteps]
      else:
        slices = [slice(None), slice(None)] + list(histSliceSum) + [slice(1)]
        h1 = vorpalUtil.getHistory(simNames[0], histNames[hi], slices)[timeSteps]
        # GXY histories had bug leaving out bin for particles exceeding
        #   max gamma...add it here to avoid error
        if h1.shape[1] < gammaBinEdges.size:
          if not suppressOutput:
            print "Warning: %s doesn't have bin for gamma exceeding max" % histNames[hi]
          hasSpilloverBin = False
        # h1 may not have number of macro-particles (in case of constant weight ptcls)
        if h1.shape[-1] < 2: #{
          # assume constant weight
          posNames = ["".join(s) for s in itertools.product(
              ("dn","up"), ("bg", "drift"), ("Electrons", "Ions", "Positrons"))]
          speciesName = None
          for spName in posNames:
            if re.match(spName, histNames[hi]):
              speciesName = spName
          if speciesName is None:
            msg = "Cannot determine which species yielded " + histNames[hi] 
            raise ValueError, msg
          sprops = vorpalUtil.getSpeciesProperties(simNames[0], speciesName)
          h1MacroNumber = h1 / sprops["macroNumber"]
          h1 = numpy.concatenate((h1, h1MacroNumber), axis=-1)
        #}
        # sum over histSliceSum
        while len(h1.shape) > 3:
          h1 = h1.sum(axis = -2)
      if hasSpilloverBin and len(h1) > 2 and h1.shape[-1]>1 and (h1[:,-1,1] > 0).any():
        if not suppressOutput:
          print "Warning:", simNames[0], ":", histNames[hi], 
          print "had particles exceeding the maximum gamma-bin."
    except:
      print histNames[hi] + ".shape =", sh
      raise
    if hasSpilloverBin:
      h += h1[:,:-1,:]
    else:
      h += h1[:,:,:]
  
  
  try:
    times = vorpalUtil.getHistory(simNames[0], "smallDistTimes")[:,0]
    if times.size != sh[0]:
      if not suppressOutput:
        print "Using largeDistTimes"
      times = vorpalUtil.getHistory(simNames[0], "largeDistTimes")[:,0]
      if times.size != sh[0]:
        if not suppressOutput:
          print "can't find right dist times"
        times = numpy.zeros(sh[0], dtype=numpy.float64)
    times = times[timeSteps]
        
  except:
    print "Warning: DistTimes not found"
    raise ValueError, "DistTimes not found"
    times = numpy.array(timeSteps)*1.86805e-9

  if writeFitsFile is None:
    gm1List = [gammaM1] if gammaM1 != "both" else [False, True]
    for gm1 in gm1List:
      addDistCharacteristicsKernel(sDict,
        h, times, gammaBinEdges, timeSteps,
        histSliceSum = histSliceSum, gammaM1 = gm1, 
        estimateFinal = estimateFinal, cachedFitData = cachedFitData)
  else:
    writeFits(writeFitsFile, sDict, h, times, gammaBinEdges)
#}

def getBasicSimParams(simName): #{
  vf = genUtil.importFullPath(vorpalUtil.getVarsName(simName))
  sDict = {
    "simName":simName,
    "GAMMA_SCALE":vf.GAMMA_SCALE,
    "rhoc":vf.LARMOR_LENGTH,
    "rho0":vf.LARMOR_LENGTH/vf.GAMMA_SCALE,
    "omegac":vf.LARMOR_FREQ,
    "omega0":vf.LARMOR_FREQ*vf.GAMMA_SCALE,
    "LX_TOT":vf.LX_TOTS[0],
    "LY_TOT":vf.LX_TOTS[1],
    "DX":vf.DX,
    "DY":vf.DY,
    "B_0":vf.B_0,
    "ndOvernb":vf.DENSITY_0/vf.DENSITY_BG,
    "thetade":vf.kB_T_S_OVER_MCSQR,
    "thetadi":vf.kB_T_S_ION_OVER_MCSQR,
    "thetabe":vf.kB_T_S_OVER_MCSQR * vf.T_BG/vf.T_S,
    "thetabi":vf.kB_T_S_ION_OVER_MCSQR * vf.T_BG_ION/vf.T_S_ION,
    "betade":vf.BETA_DRIFT,
    "betadi":vf.BETA_DRIFT_ION,
    "gammade":vf.GAMMA_DRIFT,
    "gammadi":vf.GAMMA_DRIFT_ION,
    "SMOOTH_E":vf.SMOOTH_E,
    "SMOOTH_J":vf.SMOOTH_J,
    "bgPpc":vf.BG_MPTCLS_PER_DX*vf.BG_MPTCLS_PER_DY,
    "dPpc":vf.BG_MPTCLS_PER_DX*vf.BG_MPTCLS_PER_DY,
    "bgPpc":vf.MPTCLS_PER_DX*vf.MPTCLS_PER_DY,
    "dPpc":vf.MPTCLS_PER_DX*vf.MPTCLS_PER_DY,
    "sigma": (vf.B_0**2 / mu0)/(
      vf.DENSITY_BG * (vf.ELECMASS + vf.IONMASS) * c**2),
    "sigmae": (vf.B_0**2 / mu0)/(vf.DENSITY_BG * vf.ELECMASS * c**2),
    "sigmai": (vf.B_0**2 / mu0)/(vf.DENSITY_BG * vf.IONMASS * c**2),
    "m":vf.IONMASS/vf.ELECMASS,
    "BgOverB0":vf.ALPHA,
    "DT":vf.DT,
    "SIMTIME":vf.TIMESTEPS*vf.DT,
    "DOMAIN_DECOMP":vf.DOMAIN_DECOMP,
    "problemQ":0,
    "problems":[],
    }
  try:
    vf.USE_VAY_MOVE,
    sDict["USE_VAY_MOVE"] = vf.USE_VAY_MOVE
  except:
    sDict["USE_VAY_MOVE"] = 0
  return sDict
#}

def addTiming(sDict):  #{
  simName = sDict["simName"]
  try:
    [numSteps, totalWallTime, simWallTime, numCores, numCells
      ] = vorpalUtil.graphPerformance(simName + ".out", plot = False,
        suppressOutput = True)
  except:
    raise
    [numSteps, totalWallTime, simWallTime, numCores, numCells] = [0,0,0,0,0]
  sDict["numSteps"] = numSteps
  sDict["totalWallTime"] = totalWallTime
  sDict["simWallTime"] = simWallTime
  sDict["numCores"] = int(round(numCores))
  sDict["numCells"] = numCells
#}


def simStatsStr(simName): #{
  sDict = getBasicSimParams(simName)
  addEnergyGrowthAndEndOfReconnectionTime(sDict)
  addTiming(sDict)
  addFluxReconFrac(sDict)
  ghistNames = [k + "bgElectronsGammaDist" for k in ["dn", "up"]]
  if sDict["m"] == 1.:
    ghistNames += [k + "bgIonsGammaDist" for k in ["dn", "up"]]
  wff = "bgPtclsFits.csv"
  if 1:
    addDistCharacteristics(sDict, ghistNames, 
      gammaM1 = False, writeFitsFile = wff)
  import arrayIO
  fitData = arrayIO.readArrayFromFile(wff, sep=',')
  addDistCharacteristics(sDict, ghistNames, gammaM1 = 'both',
    cachedFitData = fitData)
  s = []
  # number of std devs up and down
  nstd = 1.
  global colDescStrs
  global colVals
  colDescStrs = []
  colVals = []
  def addFromDict(descStr, key = None):
    global colDescStrs
    global colVals
    if key is None:
      key = descStr
    colDescStrs.append(descStr)
    colVals.append(sDict[key])
  def addVal(descStr, val):
    global colDescStrs
    global colVals
    colDescStrs.append(descStr)
    colVals.append(val)
  oldColStrs = ["sigma_e/2","theta_de","gamma_scale","theta_be", "L/rho_c",
    "rho_c/dx","ppc","Vay push", "smoothE", "smoothJ", 
    "#steps","time (s)","cores",
    "domDecomp","L/rho_0","pprho_c^2",
    "rho_0/dx","pprho_0^2","E variation(%)",
    "g_2 (lo)","g_2 (hi)", "alpha (lo)","alpha (hi)",
    "ggm1_2 (lo)","ggm1_2 (hi)", "alphagm1 (lo)","alphagm1 (hi)",
    "magEnergyDropFrac", "reconFluxFrac",
    "problemQ", "problems",
    ]
  addVal("sigma_e/2", sDict["sigmae"]/2.)
  addFromDict("theta_de", "thetade")
  addFromDict("gamma_scale", "GAMMA_SCALE")
  addFromDict("theta_be", "thetabe")
  L = sDict["LX_TOT"]
  rhoc = sDict["rhoc"]
  rho0 = sDict["rho0"]
  dx = sDict["DX"]
  addVal("L/rho_c", L/rhoc)
  addVal("rho_c/dx", rhoc/dx)
  ppc = sDict["bgPpc"]
  addVal("ppc", ppc)
  addFromDict("Vay push", "USE_VAY_MOVE")
  addFromDict("smoothE", "SMOOTH_E")
  addFromDict("smoothJ", "SMOOTH_J")
  addFromDict("#steps", "numSteps")
  addFromDict("time (s)", "totalWallTime")
  addFromDict("cores", "numCores")
  addFromDict("sim time (s)", "simWallTime")
  addFromDict("domDecomp", "DOMAIN_DECOMP")
  addVal("L/rho_0", L/rho0)
  addVal("pprho_c^2", ppc * (rhoc/dx)**2)
  addVal("rho_0/dx", rho0/dx)
  addVal("pprho_0^2", ppc * (rho0/dx)**2)
  addVal("E variation(%)", sDict["energyVarByReconEndPlusCrossTime"]*100.)
  addVal("g_2 (lo)", sDict["g2f"][0] - nstd * sDict["g2f"][1])
  addVal("g_2 (hi)", sDict["g2f"][0] + nstd * sDict["g2f"][1])
  addVal("alpha (lo)", sDict["powersf"][0] - nstd * sDict["powersf"][1])
  addVal("alpha (hi)", sDict["powersf"][0] + nstd * sDict["powersf"][1])
  addVal("alphaLongest (lo)", sDict["longestPowersf"][0] - nstd * sDict["longestPowersf"][1])
  addVal("alphaLongest (hi)", sDict["longestPowersf"][0] + nstd * sDict["longestPowersf"][1])
  addVal("ggm1_2 (lo)", sDict["g2fgm1"][0] - nstd * sDict["g2fgm1"][1])
  addVal("ggm1_2 (hi)", sDict["g2fgm1"][0] + nstd * sDict["g2fgm1"][1])
  addVal("alphagm1 (lo)", sDict["powersfgm1"][0] - nstd * sDict["powersfgm1"][1])
  addVal("alphagm1 (hi)", sDict["powersfgm1"][0] + nstd * sDict["powersfgm1"][1])
  addVal("alphaLongestgm1 (lo)", sDict["longestPowersfgm1"][0] - nstd * sDict["longestPowersfgm1"][1])
  addVal("alphaLongestgm1 (hi)", sDict["longestPowersfgm1"][0] + nstd * sDict["longestPowersfgm1"][1])
  addFromDict("magEnergyDropFrac", "magEnergyDropFrac")
  addFromDict("reconFluxFrac", "reconnectedFluxFrac")
  addFromDict("distCharEndOfReconTime")
  addFromDict("distCharEndOfReconTimePlus")
  #addFromDict("thetaLowE")
  #addFromDict("thetaLowEFail")
  #addFromDict("fracInMaxLowE")
  #addFromDict("energyFracInMaxLowE")
  #addFromDict("g_aboveLowEdist", "gAfterMaxLowE")
    # theta_{low energy Maxwellian} = thGrowth[1] * exp(thGrowth[0] * t)
  addVal("thetaLowEgrowthRate", sDict["thetaLowEgrowth"][1])
  addVal("thetaLowEgrowthInit", sDict["thetaLowEgrowth"][0])
    
  addFromDict("fitPowersf")
  addFromDict("fitPowerse")
  addFromDict("fitPowerseStddev")
  addFromDict("fitb1sf")
  addFromDict("fitb2sf")
  addFromDict("fitg1peaksf")
  addFromDict("fitg2peaksf")
  addFromDict("fitg3peaksf")
  addFromDict("fitFracInMaxLowEf")
  addFromDict("fitThetaLowE1f")
  addFromDict("fitThetaLowE2f")

  addFromDict("distCharEndOfReconTimeStep")
  addFromDict("distCharEndOfReconTimePlusStep")
  addFromDict("problemQ", "problemQ")
  addFromDict("problems", "problems")

  # this will be a csv-file, so replace any commas within an entry with ";"
  s = [string.replace(str(cv), ',', ';') for cv in colVals]

  # For some reason, alphagm1 (lo) - alphaLongestgm1 (hi) have semicolons
  # between them instead of commas?

  #omegac = sDict["omegac"]
  #print sDict["tReconEnd"]*omegac,  sDict["tReconEndPlusCrossTime"]*omegac
  return (','.join(s), ','.join(colDescStrs))
#}

if __name__ == "__main__":
  suppressOutput = True
  s = simStatsStr("relRecon2p")
  print s[1]
  print s[0]
