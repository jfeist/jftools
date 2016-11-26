# coding: utf-8

import scipy.linalg
import numpy as np
import math

def simpsrule(n):
    if n%2==0:
        raise ValueError('n must be odd for simpson rule')
    if n <= 1:
        t = np.zeros(1)
        w = 2.*np.ones(1)
    else:
        t = np.linspace(-1,1,n)
        h = t[1]-t[0]
        w = np.ones(n)
        w[1:-1:2] = 4
        w[2:-2:2] = 2
        w *= h/3
    return t,w

def classpol(ikind, n, alpha=0., beta=0.):
    # this procedure supplies the coefficients a(j), b(j) of the
    # recurrence relation
    #   b p (x) = (x - a ) p   (x) - b   p   (x)
    #    j j            j   j-1       j-1 j-2
    # for the various classical (normalized) orthogonal polynomials,
    # and the zero-th moment
    #   muzero = integral w(x) dx
    # of the given polynomial weight function w(x). since the
    # polynomials are orthonormalized, the tridiagonal matrix is
    # guaranteed to be symmetric.
    # the input parameter alpha is used only for laguerre and
    # jacobi polynomials, and the parameter beta is used only for
    # jacobi polynomials.  the laguerre and jacobi polynomials
    # require the gamma function.
    if ikind==1:
        # ikind = 1=  legendre polynomials p(x)
        # on (-1, +1), w(x) = 1.
        muzero = 2.
        a = np.zeros(n)
        iis = np.arange(1,n)
        b = iis/np.sqrt(4*iis**2-1)
    elif ikind==2:
        # ikind = 2=  chebyshev polynomials of the first ikind t(x)
        # on (-1, +1), w(x) = 1 / sqrt(1 - x*x)
        muzero = np.pi
        a = np.zeros(n)
        b = 0.5*np.ones(n-1)
        b[0] = np.sqrt(0.5)
    elif ikind==3:
        # ikind = 3=  chebyshev polynomials of the second ikind u(x)
        # on (-1, +1), w(x) = sqrt(1 - x*x)
        muzero = np.pi/2.
        a = np.zeros(n)
        b = 0.5*np.ones(n-1)
    elif ikind==4:
        # ikind = 4=  hermite polynomials h(x)
        # on (-infinity,+infinity), w(x) = exp(-x**2)
        muzero = np.sqrt(np.pi)
        a = np.zeros(n)
        b = np.sqrt(np.arange(1,n)/2)
    elif ikind==5:
        # ikind = 5=  jacobi polynomials p(alpha, beta)(x)
        # on (-1, +1), w(x) = (1-x)**alpha + (1+x)**beta,
        # alpha and beta greater than -1
        alpha = float(alpha)
        beta = float(beta)
        ab = alpha + beta
        abi = 2.+ab
        muzero = 2.**(ab+1) * math.gamma(alpha+1) * math.gamma(beta+1.) / math.gamma(abi)
        a = np.empty(n)
        b = np.empty(n-1)
        a[0] = (beta - alpha)/abi
        b[0] = np.sqrt(4.*(1.+alpha)*(1.+beta)/((abi+1.)*abi**2))
        a2b2 = beta**2 - alpha**2
        for ii in range(1,n-1):
            jj = ii + 1
            abi = 2.*jj + ab
            a[ii] = a2b2/((abi-2.)*abi)
            b[ii] = np.sqrt(4.*jj*(jj+alpha)*(jj+beta)*(jj+ab)/((abi**2-1)*abi**2))
        abi = 2.*n + ab
        a[-1] = a2b2/((abi-2.)*abi)
    elif ikind==6:
        # ikind = 6=  laguerre polynomials l(alpha)(x)
        # on (0, +infinity), w(x) = exp(-x) * x**alpha, alpha greater than -1.
        alpha = float(alpha)
        muzero = math.gamma(alpha+1.)
        iis = np.arange(1,n+1)
        a = 2*iis - 1 + alpha
        b = np.sqrt(iis[:-1]*(iis[:-1]+alpha))
    return b, a, muzero

def gbslve(shift, a, b):
    """this procedure performs elimination to solve for the
    n-th component of the solution delta to the equation
         (jn - shift*identity) * delta  = en,
    where en is the vector of all zeroes except for 1 in
    the n-th position.
    the matrix jn is symmetric tridiagonal, with diagonal
    elements a(i), off-diagonal elements b(i).  this equation
    must be solved to obtain the appropriate changes in the lower
    2 by 2 submatrix of coefficients for orthogonal polynomials."""
    alpha = a[0] - shift
    for ii in range(1,len(a)-1):
        alpha = a[ii] - shift - b[ii-1]**2/alpha
    return 1./alpha

def gaussq(kind, n, endpts, alpha=0., beta=0.):
    #        this set of routines computes the nodes x(i) and weights
    #        c(i) for gaussian-type quadrature rules with pre-assigned
    #        nodes.  these are used when one wishes to approximate

    #                 integral (from a to b)  f(x) w(x) dx

    #                              n
    #        by                   sum c  f(x )
    #                             i=1  i    i

    #        here w(x) is one of six possible non-negative weight
    #        functions (listed below), and f(x) is the
    #        function to be integrated.  gaussian quadrature is particularly
    #        useful on infinite intervals (with appropriate weight
    #        functions), since then other techniques often fail.

    #           associated with each weight function w(x) is a set of
    #        orthogonal polynomials.  the nodes x(i) are just the zeroes
    #        of the proper n-th degree polynomial.

    #     input parameters

    #        ikind     an integer between 0 and 6 giving the type of
    #                 quadrature rule

    #        ikind = 0=  simpson's rule w(x) = 1 on (-1, 1) n must be odd.
    #        ikind = 1=  legendre quadrature, w(x) = 1 on (-1, 1)
    #        ikind = 2=  chebyshev quadrature of the first ikind
    #                   w(x) = 1/dsqrt(1 - x*x) on (-1, +1)
    #        ikind = 3=  chebyshev quadrature of the second ikind
    #                   w(x) = dsqrt(1 - x*x) on (-1, 1)
    #        ikind = 4=  hermite quadrature, w(x) = exp(-x*x) on
    #                   (-infinity, +infinity)
    #        ikind = 5=  jacobi quadrature, w(x) = (1-x)**alpha * (1+x)**
    #                   beta on (-1, 1), alpha, beta .gt. -1.
    #                   note= ikind=2 and 3 are a special case of this.
    #        ikind = 6=  generalized laguerre quadrature, w(x) = exp(-x)*
    #                   x**alpha on (0, +infinity), alpha .gt. -1

    #        n        the number of points used for the quadrature rule
    #        alpha    real(doub_prec) parameter used only for gauss-jacobi and gauss-
    #                 laguerre quadrature (otherwise use 0.).
    #        beta     real(doub_prec) parameter used only for gauss-jacobi quadrature--
    #                 (otherwise use 0.).
    #        kpts     (integer) normally 0, unless the left or right end-
    #                 point (or both) of the interval is required to be a
    #                 node (this is called gauss-radau or gauss-lobatto
    #                 quadrature).  then kpts is the number of fixed
    #                 endpoints (1 or 2).
    #        endpts   real(doub_prec) array of length 2.  contains the values of
    #                 any fixed endpoints, if kpts = 1 or 2.
    #        b        real(doub_prec) scratch array of length n

    #     output parameters (both arrays of length n)

    #        t        will contain the desired nodes x(1),,,x(n)
    #        w        will contain the desired weights c(1),,,c(n)

    #     subroutines required

    #        gbslve, class, and gbtql2 are provided. underflow may sometimes
    #        occur, but it is harmless if the underflow interrupts are
    #        turned off as they are on this machine.

    #     accuracy

    #        the routine was tested up to n = 512 for legendre quadrature,
    #        up to n = 136 for hermite, up to n = 68 for laguerre, and up
    #        to n = 10 or 20 in other cases.  in all but two instances,
    #        comparison with tables in ref. 3 showed 12 or more significant
    #        digits of accuracy.  the two exceptions were the weights for
    #        hermite and laguerre quadrature, where underflow caused some
    #        very small weights to be set to zero.  this is, of course,
    #        completely harmless.

    #     method

    #           the coefficients of the three-term recurrence relation
    #        for the corresponding set of orthogonal polynomials are
    #        used to form a symmetric tridiagonal matrix, whose
    #        eigenvalues (determined by the implicit ql-method with
    #        shifts) are just the desired nodes.  the first components of
    #        the orthonormalized eigenvectors, when properly scaled,
    #        yield the weights.  this technique is much faster than using a
    #        root-finder to locate the zeroes of the orthogonal polynomial.
    #        for further details, see ref. 1.  ref. 2 contains details of
    #        gauss-radau and gauss-lobatto quadrature only.

    #     references

    #        1.  golub, g. h., and welsch, j. h.,  calculation of gaussian
    #            quadrature rules,  mathematics of computation 23 (april,
    #            1969), pp. 221-230.
    #        2.  golub, g. h.,  some modified matrix eigenvalue problems,
    #            siam review 15 (april, 1973), pp. 318-334 (section 7).
    #        3.  stroud and secrest, gaussian quadrature formulas, prentice-
    #            hall, englewood cliffs, n.j., 1966.

    #     ..................................................................
    kinds = ('simpson','legendre','chebyshev-1','chebyshev-2','hermite','jacobi','laguerre')
    ikind = kinds.index(kind)
    
    if ikind==0:
        return simpsrule(n)

    b, t, muzero = classpol(ikind, n, alpha, beta)
    # the matrix of coefficients is assumed to be symmetric.
    # the array t contains the diagonal elements, the array
    # b the off-diagonal elements.
    # make appropriate changes in the lower right 2 by 2
    # submatrix.

    if len(endpts)==1:
        # if kpts=1, only t(n) must be changed
        t[-1] = gbslve(endpts[0], t, b) * b[-1]**2 + endpts[0]
    elif len(endpts)==2:
        # if kpts=2, t(n) and b(n-1) must be recomputed
        gam = gbslve(endpts[0], t, b)
        t1 = (endpts[0] - endpts[1]) / (gbslve(endpts[1], t, b) - gam)
        b[-1] = np.sqrt(t1)
        t[-1] = endpts[0] + gam*t1

    # now compute the eigenvalues of the symmetric tridiagonal
    # matrix, which has been modified as necessary.
    # the method used is a ql-type method with origin shifting
    
    # upper form:
    # *   *   a02 a13 a24 a35
    # *   a01 a12 a23 a34 a45
    # a00 a11 a22 a33 a44 a55
    A = np.empty((2,n))
    A[0,1:] = b
    A[1,:]  = t
    t, w = scipy.linalg.eig_banded(A)
    w = muzero * w[0,:]**2
    return t, w


def lgngr(x,y):
    """Finds Lagrange interpolating polynomials of function of x 
and their first and second derivatives on an arbitrary grid y."""
    from itertools import chain
    nx, ny = len(x), len(y)
    p   = np.empty((ny,nx))
    dp  = np.empty_like(p)
    ddp = np.empty_like(p)
    dxx = x[:,None] - x[None,:]
    dyx = y[:,None] - x[None,:]
    #     generate polynomials and derivatives with respect to x
    for i in range(ny):
        zerfac = -1
        for j in range(nx):
            if (abs(dyx[i,j]) <= 1e-10):
                zerfac = j
        for j in range(nx):
            p[i,j] = 1.
            for k in chain(range(j),range(j+1,nx)):
                p[i,j] *= dyx[i,k] / dxx[j,k]
            if abs(p[i,j])>1e-10:
                sn = 0.
                ssn = 0.
                for k in chain(range(j),range(j+1,nx)):
                    fac = 1./dyx[i,k]
                    sn += fac
                    ssn += fac**2
                dp[i,j] = sn*p[i,j]
                ddp[i,j] = sn*dp[i,j] - ssn*p[i,j]
            else:
                sn = 1.
                ssn = 0.
                for k in range(nx):
                    if k in (j,zerfac):
                        continue
                    fac = 1./dxx[j,k]
                    sn *= fac*dyx[i,k]
                    ssn += 1./dyx[i,k]
                dp[i,j] = sn/dxx[j,zerfac]
                ddp[i,j] = 2.*ssn*dp[i,j]
    return p,dp,ddp

def dvr_reg(nfun,xbounds):
    # Find zeros of nfun'th order Legendre quadrature (Gauss-Lobatto)
    pt, wt = gaussq('legendre',nfun,[-1,1])

    # Set up rescaled position grid
    xmin, xmax = xbounds
    A = abs(xmax-xmin)/2.
    B = (xmax+xmin)/2.
    pt = A*pt + B
    wt = A*wt

    # Generate Lagrange interpolating polynomials 
    # and their first and second derivatives
    f,df,ddf = lgngr(pt,pt)

    # Set up kinetic energy matrix
    # ke[function index, point index]
    ke_mat = ddf * wt[:,None]
    # add the bloch contributions
    ke_mat[ 0,:] +=  f[ 0, 0]*df[ 0,:]
    ke_mat[-1,:] += -f[-1,-1]*df[-1,:]
    ke_mat *= -0.5
    return pt,wt,df,ddf,ke_mat
