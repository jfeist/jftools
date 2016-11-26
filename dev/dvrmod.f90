module kinddefs
  integer, parameter :: doub_prec = kind(1.d0)
end module kinddefs
MODULE helpers
  use kinddefs
contains
  !deck @(#)gaussq.f 1.1 9/9/91
  ! Code converted using TO_F90 by Alan Miller
  ! Date: 2000-11-02  Time: 11:45:58
  SUBROUTINE gaussq(ckind, n, alpha, beta, kpts, endpts, b, t, w)
    !           this set of routines computes the nodes x(i) and weights
    !        c(i) for gaussian-type quadrature rules with pre-assigned
    !        nodes.  these are used when one wishes to approximate

    !                 integral (from a to b)  f(x) w(x) dx

    !                              n
    !        by                   sum c  f(x )
    !                             i=1  i    i

    !        here w(x) is one of six possible non-negative weight
    !        functions (listed below), and f(x) is the
    !        function to be integrated.  gaussian quadrature is particularly
    !        useful on infinite intervals (with appropriate weight
    !        functions), since then other techniques often fail.

    !           associated with each weight function w(x) is a set of
    !        orthogonal polynomials.  the nodes x(i) are just the zeroes
    !        of the proper n-th degree polynomial.

    !     input parameters

    !        ikind     an integer between 0 and 6 giving the type of
    !                 quadrature rule

    !        ikind = 0=  simpson's rule w(x) = 1 on (-1, 1) n must be odd.
    !        ikind = 1=  legendre quadrature, w(x) = 1 on (-1, 1)
    !        ikind = 2=  chebyshev quadrature of the first ikind
    !                   w(x) = 1/dsqrt(1 - x*x) on (-1, +1)
    !        ikind = 3=  chebyshev quadrature of the second ikind
    !                   w(x) = dsqrt(1 - x*x) on (-1, 1)
    !        ikind = 4=  hermite quadrature, w(x) = exp(-x*x) on
    !                   (-infinity, +infinity)
    !        ikind = 5=  jacobi quadrature, w(x) = (1-x)**alpha * (1+x)**
    !                   beta on (-1, 1), alpha, beta .gt. -1.
    !                   note= ikind=2 and 3 are a special case of this.
    !        ikind = 6=  generalized laguerre quadrature, w(x) = exp(-x)*
    !                   x**alpha on (0, +infinity), alpha .gt. -1

    !        n        the number of points used for the quadrature rule
    !        alpha    real(doub_prec) parameter used only for gauss-jacobi and gauss-
    !                 laguerre quadrature (otherwise use 0.).
    !        beta     real(doub_prec) parameter used only for gauss-jacobi quadrature--
    !                 (otherwise use 0.).
    !        kpts     (integer) normally 0, unless the left or right end-
    !                 point (or both) of the interval is required to be a
    !                 node (this is called gauss-radau or gauss-lobatto
    !                 quadrature).  then kpts is the number of fixed
    !                 endpoints (1 or 2).
    !        endpts   real(doub_prec) array of length 2.  contains the values of
    !                 any fixed endpoints, if kpts = 1 or 2.
    !        b        real(doub_prec) scratch array of length n

    !     output parameters (both arrays of length n)

    !        t        will contain the desired nodes x(1),,,x(n)
    !        w        will contain the desired weights c(1),,,c(n)

    !     subroutines required

    !        gbslve, classpol, and gbtql2 are provided. underflow may sometimes
    !        occur, but it is harmless if the underflow interrupts are
    !        turned off as they are on this machine.

    !     accuracy

    !        the routine was tested up to n = 512 for legendre quadrature,
    !        up to n = 136 for hermite, up to n = 68 for laguerre, and up
    !        to n = 10 or 20 in other cases.  in all but two instances,
    !        comparison with tables in ref. 3 showed 12 or more significant
    !        digits of accuracy.  the two exceptions were the weights for
    !        hermite and laguerre quadrature, where underflow caused some
    !        very small weights to be set to zero.  this is, of course,
    !        completely harmless.

    !     method

    !           the coefficients of the three-term recurrence relation
    !        for the corresponding set of orthogonal polynomials are
    !        used to form a symmetric tridiagonal matrix, whose
    !        eigenvalues (determined by the implicit ql-method with
    !        shifts) are just the desired nodes.  the first components of
    !        the orthonormalized eigenvectors, when properly scaled,
    !        yield the weights.  this technique is much faster than using a
    !        root-finder to locate the zeroes of the orthogonal polynomial.
    !        for further details, see ref. 1.  ref. 2 contains details of
    !        gauss-radau and gauss-lobatto quadrature only.

    !     references

    !        1.  golub, g. h., and welsch, j. h.,  calculation of gaussian
    !            quadrature rules,  mathematics of computation 23 (april,
    !            1969), pp. 221-230.
    !        2.  golub, g. h.,  some modified matrix eigenvalue problems,
    !            siam review 15 (april, 1973), pp. 318-334 (section 7).
    !        3.  stroud and secrest, gaussian quadrature formulas, prentice-
    !            hall, englewood cliffs, n.j., 1966.

    !     ..................................................................


    IMPLICIT REAL(DOUB_PREC) (a-h,o-z)
    CHARACTER (LEN=*), INTENT(IN)            :: ckind
    INTEGER, INTENT(IN)                      :: n
    REAL(DOUB_PREC), INTENT(IN)                     :: alpha
    REAL(DOUB_PREC), INTENT(IN)                     :: beta
    INTEGER, INTENT(IN)                      :: kpts
    REAL(DOUB_PREC), INTENT(IN)                         :: endpts(2)
    REAL(DOUB_PREC), INTENT(IN OUT)                     :: b(n)
    REAL(DOUB_PREC), INTENT(OUT)                        :: t(n)
    REAL(DOUB_PREC), INTENT(OUT)                        :: w(n)
    REAL(DOUB_PREC) :: muzero

    IF (ckind == 'simpson') THEN
       ikind=0
    ELSE IF(ckind == 'legendre') THEN
       ikind=1
    ELSE IF(ckind == 'chebyshev-1') THEN
       ikind=2
    ELSE IF(ckind == 'chebyshev-2') THEN
       ikind=3
    ELSE IF(ckind == 'hermite') THEN
       ikind=4
    ELSE IF(ckind == 'jacobi') THEN
       ikind=5
    ELSE IF(ckind == 'laguerre') THEN
       ikind=6
    ELSE
       stop 
       !  CALL lnkerr('error in quadrature type')
    END IF
    IF(ikind == 0) THEN
       IF(2*(n/2) == n) THEN
          stop
          !    CALL lnkerr('n must be odd for simpson rule')
       END IF
       IF(n <= 1) THEN
          t(1) = 0.d+00
          w(1) = 2.d+00
          RETURN
       END IF
       h = 2.d+00/(n-1)
       t(1) = -1.d+00
       t(n) = 1.d+00
       w(1) = h/3.d+00
       w(n) = h/3.d+00
       nm1 = n-1
       DO  i=2,nm1
          t(i) = t(i-1) + h
          w(i) = 4.d+00 - 2.d+00*(i-2*(i/2))
          w(i) = w(i)*h/3.d+00
       END DO
       RETURN
    END IF

    CALL classpol (ikind, n, alpha, beta, b, t, muzero)

    !           the matrix of coefficients is assumed to be symmetric.
    !           the array t contains the diagonal elements, the array
    !           b the off-diagonal elements.
    !           make appropriate changes in the lower right 2 by 2
    !           submatrix.

    IF (kpts == 0)  GO TO 100
    IF (kpts == 2)  GO TO  50

    !           if kpts=1, only t(n) must be changed

    t(n) =gbslve(endpts(1), n, t, b)*b(n-1)**2 + endpts(1)
    GO TO 100

    !           if kpts=2, t(n) and b(n-1) must be recomputed

50  gam =gbslve(endpts(1), n, t, b)
    t1 = ((endpts(1) - endpts(2))/(gbslve(endpts(2), n, t, b) - gam))
    b(n-1) =  SQRT(t1)
    t(n) = endpts(1) + gam*t1

    !           note that the indices of the elements of b run from 1 to n-1
    !           and thus the value of b(n) is arbitrary.
    !           now compute the eigenvalues of the symmetric tridiagonal
    !           matrix, which has been modified as necessary.
    !           the method used is a ql-type method with origin shifting

100 w(1) = 1.0D0
    DO  i = 2, n
       w(i) = 0.0D0
    END DO

    CALL gbtql2 (n, t, b, w, ierr)
    DO  i = 1, n
       w(i) = muzero * w(i) * w(i)
    END DO

    RETURN
  END SUBROUTINE gaussq


  !deck @(#)classpol.f 1.1 9/9/91
  ! Code converted using TO_F90 by Alan Miller
  ! Date: 2000-11-02  Time: 11:46:10
  SUBROUTINE classpol(ikind, n, alpha, beta, b, a, muzero)

    !           this procedure supplies the coefficients a(j), b(j) of the
    !        recurrence relation

    !             b p (x) = (x - a ) p   (x) - b   p   (x)
    !              j j            j   j-1       j-1 j-2

    !        for the various classical (normalized) orthogonal polynomials,
    !        and the zero-th moment

    !             muzero = integral w(x) dx

    !        of the given polynomial   weight function w(x).  since the
    !        polynomials are orthonormalized, the tridiagonal matrix is
    !        guaranteed to be symmetric.

    !           the input parameter alpha is used only for laguerre and
    !        jacobi polynomials, and the parameter beta is used only for
    !        jacobi polynomials.  the laguerre and jacobi polynomials
    !        require the gamma function.

    !     ..................................................................


    IMPLICIT REAL(DOUB_PREC) (a-h,o-z)
    INTEGER, INTENT(IN OUT)                  :: ikind
    INTEGER, INTENT(IN)                      :: n
    REAL(DOUB_PREC), INTENT(IN)                         :: alpha
    REAL(DOUB_PREC), INTENT(IN)                         :: beta
    REAL(DOUB_PREC), INTENT(OUT)                        :: b(n)
    REAL(DOUB_PREC), INTENT(OUT)                        :: a(n)
    REAL(DOUB_PREC), INTENT(OUT)                      :: muzero

    DATA pi / 3.141592653589793D0  /

    nm1 = n - 1
    SELECT CASE ( ikind )
    CASE (    1)
       GO TO 10
    CASE (    2)
       GO TO  20
    CASE (    3)
       GO TO  30
    CASE (    4)
       GO TO  40
    CASE (    5)
       GO TO  50
    CASE (    6)
       GO TO  60
    END SELECT

    !              ikind = 1=  legendre polynomials p(x)
    !              on (-1, +1), w(x) = 1.

10  muzero = 2.0D0
    DO  i = 1, nm1
       a(i) = 0.0D0
       abi = i
       b(i) = abi/ SQRT(4.d0*abi*abi - 1.0D0  )
    END DO
    a(n) = 0.0D0
    RETURN

    !              ikind = 2=  chebyshev polynomials of the first ikind t(x)
    !              on (-1, +1), w(x) = 1 / sqrt(1 - x*x)

20  muzero = pi
    DO  i = 1, nm1
       a(i) = 0.0D0
       b(i) = 0.5D0
    END DO
    b(1) =  SQRT(0.5D0  )
    a(n) = 0.0D0
    RETURN

    !              ikind = 3=  chebyshev polynomials of the second ikind u(x)
    !              on (-1, +1), w(x) = sqrt(1 - x*x)

30  muzero = pi/2.0D0
    DO  i = 1, nm1
       a(i) = 0.0D0
       b(i) = 0.5D0
    END DO
    a(n) = 0.0D0
    RETURN

    !              ikind = 4=  hermite polynomials h(x) on (-infinity,
    !              +infinity), w(x) = exp(-x**2)

40  muzero =  SQRT(pi)
    DO  i = 1, nm1
       a(i) = 0.0D0
       b(i) =  SQRT(i/2.0D0  )
    END DO
    a(n) = 0.0D0
    RETURN

    !              ikind = 5=  jacobi polynomials p(alpha, beta)(x) on
    !              (-1, +1), w(x) = (1-x)**alpha + (1+x)**beta, alpha and
    !              beta greater than -1

50  ab = alpha + beta
    abi = 2.0D0   + ab
    muzero = 2.0D0   ** (ab + 1.0D0  ) * gamma(alpha + 1.0D0  ) * gamma(  &
         beta + 1.0D0  ) / gamma(abi)
    a(1) = (beta - alpha)/abi
    b(1) =  SQRT(4.0D0  *(1.0D0  + alpha)*(1.0D0   + beta)/((abi + 1.0D0  )*  &
         abi*abi))
    a2b2 = beta*beta - alpha*alpha
    DO  i = 2, nm1
       abi = 2.0D0  *i + ab
       a(i) = a2b2/((abi - 2.0D0  )*abi)
       b(i) =  SQRT (4.0D0  *i*(i + alpha)*(i + beta)*(i + ab)/  &
            ((abi*abi - 1)*abi*abi))
    END DO
    abi = 2.0D0  *n + ab
    a(n) = a2b2/((abi - 2.0D0  )*abi)
    RETURN

    !              ikind = 6=  laguerre polynomials l(alpha)(x) on
    !              (0, +infinity), w(x) = exp(-x) * x**alpha, alpha greater
    !              than -1.

60  muzero = gamma(alpha + 1.0D0)
    DO  i = 1, nm1
       a(i) = 2.0D0  *i - 1.0D0   + alpha
       b(i) =  SQRT(i*(i + alpha))
    END DO
    a(n) = 2.0D0  *n - 1 + alpha
    RETURN
  END SUBROUTINE classpol


  !deck @(#)gbslve.f 1.1 9/9/91
  ! Code converted using TO_F90 by Alan Miller
  ! Date: 2000-11-02  Time: 11:46:27
  FUNCTION gbslve(shift, n, a, b)
    !       this procedure performs elimination to solve for the
    !       n-th component of the solution delta to the equation

    !             (jn - shift*identity) * delta  = en,

    !       where en is the vector of all zeroes except for 1 in
    !       the n-th position.

    !       the matrix jn is symmetric tridiagonal, with diagonal
    !       elements a(i), off-diagonal elements b(i).  this equation
    !       must be solved to obtain the appropriate changes in the lower
    !       2 by 2 submatrix of coefficients for orthogonal polynomials.

    IMPLICIT REAL(DOUB_PREC) (a-h,o-z)
    REAL(DOUB_PREC), INTENT(IN)                         :: shift
    INTEGER, INTENT(IN)                        :: n
    REAL(DOUB_PREC), INTENT(IN)                         :: a(n)
    REAL(DOUB_PREC), INTENT(IN)                         :: b(n)


    alpha = a(1) - shift
    nm1 = n - 1
    DO  i = 2, nm1
       alpha = a(i) - shift - b(i-1)**2/alpha
    END DO
    gbslve = 1.0D0  /alpha
    RETURN
  END FUNCTION gbslve

  !deck @(#)gbtql2.f 1.1 9/9/91

  ! Code converted using TO_F90 by Alan Miller
  ! Date: 2000-11-02  Time: 11:46:57

  SUBROUTINE gbtql2(n, d, e, z, ierr)

    !     this subroutine is a translation of the algol procedure imtql2,
    !     num. math. 12, 377-383(1968) by martin and wilkinson,
    !     as modified in num. math. 15, 450(1970) by dubrulle.
    !     handbook for auto. comp., vol.ii-linear algebra, 241-248(1971).

    !     this subroutine finds the eigenvalues and first components of the
    !     eigenvectors of a symmetric tridiagonal matrix by the implicit ql
    !     method, and is adapted from the eispak routine imtql2

    !     on input=

    !        n is the order of the matrix;

    !        d contains the diagonal elements of the input matrix;

    !        e contains the subdiagonal elements of the input matrix
    !          in its first n-1 positions.  e(n) is arbitrary;

    !        z contains the first row of the identity matrix.

    !      on output=

    !        d contains the eigenvalues in ascending order.  if an
    !          error exit is made, the eigenvalues are correct but
    !          unordered for indices 1, 2, ..., ierr-1;

    !        e has been destroyed;

    !        z contains the first components of the orthonormal eigenvectors
    !          of the symmetric tridiagonal matrix.  if an error exit is
    !          made, z contains the eigenvectors associated with the stored
    !          eigenvalues;

    !        ierr is set to

    !        ierr is set to
    !          zero       for normal return,
    !          j          if the j-th eigenvalue has not been
    !                     determined after 30 iterations.

    !     ------------------------------------------------------------------


    IMPLICIT REAL(DOUB_PREC) (a-h,o-z)
    INTEGER, INTENT(IN)                      :: n
    REAL(DOUB_PREC), INTENT(IN OUT)                     :: d(n)
    REAL(DOUB_PREC), INTENT(IN OUT)                     :: e(n)
    REAL(DOUB_PREC), INTENT(IN OUT)                     :: z(n)
    INTEGER, INTENT(OUT)                     :: ierr
    INTEGER :: i, j, k, l, m, ii, mml
    REAL(DOUB_PREC)  machep

    !     ========== machep is a machine dependent parameter specifying
    !                the relative precision of floating point arithmetic.
    !                machep = 16.0d0**(-13) for long form arithmetic
    !                on s360 ==========
    machep=1.0E-14

    ierr = 0
    IF (n == 1) GO TO 1001

    e(n) = 0.0D0
    DO  l = 1, n
       j = 0
       !     ========== look for small sub-diagonal element ==========
105    DO  m = l, n
          IF (m == n) GO TO 120
          IF ( ABS(e(m)) <= machep * ( ABS(d(m)) +  ABS(d(m+1)))) GO TO 120
       END DO

120    p = d(l)
       IF (m == l) CYCLE
       IF (j == 30) GO TO 1000
       j = j + 1
       !     ========== form shift ==========
       g = (d(l+1) - p) / (2.0D0   * e(l))
       r =  SQRT(g*g+1.0D0  )
       g = d(m) - p + e(l) / (g +  SIGN(r, g))
       s = 1.0D0
       c = 1.0D0
       p = 0.0D0
       mml = m - l
       !     ========== for i=m-1 step -1 until l do -- ==========
       DO  ii = 1, mml
          i = m - ii
          f = s * e(i)
          b = c * e(i)
          IF ( ABS(f) <  ABS(g)) GO TO 150
          c = g / f
          r =  SQRT(c*c+1.0D0  )
          e(i+1) = f * r
          s = 1.0D0   / r
          c = c * s
          GO TO 160
150       s = f / g
          r =  SQRT(s*s+1.0D0  )
          e(i+1) = g * r
          c = 1.0D0   / r
          s = s * c
160       g = d(i+1) - p
          r = (d(i) - g) * s + 2.0D0   * c * b
          p = s * r
          d(i+1) = g + p
          g = c * r - b
          !     ========== form first component of vector ==========
          f = z(i+1)
          z(i+1) = s * z(i) + c * f
          z(i) = c * z(i) - s * f

       END DO

       d(l) = d(l) - p
       e(l) = g
       e(m) = 0.0D0
       GO TO 105
    END DO
    !     ========== order eigenvalues and eigenvectors ==========
    DO  ii = 2, n
       i = ii - 1
       k = i
       p = d(i)

       DO  j = ii, n
          IF (d(j) >= p) CYCLE
          k = j
          p = d(j)
       END DO

       IF (k == i) CYCLE
       d(k) = d(i)
       d(i) = p

       p = z(i)
       z(i) = z(k)
       z(k) = p

    END DO

    GO TO 1001
    !     ========== set error -- no convergence to an
    !                eigenvalue after 30 iterations ==========
1000 ierr = l
1001 RETURN
    !     ========== last card of gbtql2 ==========
  END SUBROUTINE gbtql2

  
  subroutine lnkerr(message)
    CHARACTER (LEN=*), INTENT(IN) :: message
    write(6,*) message
    stop
  end subroutine lnkerr

  !============================================================
  ! Finds Lagrange interpolating polynomials
  ! of function of x and their first and second 
  ! derivatives on an arbitrary grid y.
  !============================================================
  !deck lgngr
  ! Code converted using TO_F90 by Alan Miller
  ! Date: 2000-12-06  Time: 11:11:54
  SUBROUTINE lgngr(p,dp,ddp,x,y,nx,ny,drctv)
    !***begin prologue     lgrply
    !***date written       940504   (yymmdd)
    !***revision date               (yymmdd)
    !***keywords
    !***author             schneider, barry (nsf)
    !***source             %W% %G%
    !***purpose            lagrange polynomials at arbitrary points.
    !***description
    !***

    !***references

    !***routines called

    !***end prologue       lgngr

    IMPLICIT INTEGER (a-z)
    REAL(DOUB_PREC), INTENT(OUT)                      :: p(ny,nx)
    REAL(DOUB_PREC), INTENT(OUT)                      :: dp(ny,nx)
    REAL(DOUB_PREC), INTENT(OUT)                      :: ddp(ny,nx)
    REAL(DOUB_PREC), INTENT(IN)                       :: x(nx)
    REAL(DOUB_PREC), INTENT(IN)                       :: y(ny)
    INTEGER, INTENT(IN)                      :: nx
    INTEGER, INTENT(IN)                      :: ny
    CHARACTER (LEN=*), INTENT(IN)            :: drctv


    REAL(DOUB_PREC) sn, ssn, fac

    !     generate polynomials and derivatives with respect to x

    DO  i=1,ny
       zerfac = 0
       DO  j=1,nx
          fac =  y(i) - x(j)
          IF(ABS(fac) <= 1.d-10) THEN
             zerfac = j
          END IF
       END DO
       DO  j=1,nx
          p(i,j) = 1.d0
          DO  k=1,j-1
             p(i,j) = p(i,j)*( y(i) - x(k) ) / ( x(j) - x(k) )
          END DO
          DO  k=j+1,nx
             p(i,j) = p(i,j)*( y(i) - x(k) ) / ( x(j) - x(k) )
          END DO
          IF(drctv /= 'functions-only') THEN
             IF(ABS(p(i,j)) > 1.d-10) THEN
                sn = 0.d0
                ssn = 0.d0
                DO  k=1,j-1
                   fac = 1.d0/( y(i) - x(k) )
                   sn = sn + fac
                   ssn = ssn + fac*fac
                END DO
                DO  k=j+1,nx
                   fac = 1.d0/( y(i) - x(k) )
                   sn = sn + fac
                   ssn = ssn + fac*fac
                END DO
                dp(i,j) = sn*p(i,j)
                ddp(i,j) = sn*dp(i,j) - ssn*p(i,j)
             ELSE
                first=j
                second=zerfac
                IF(first > second) THEN
                   first=zerfac
                   second=j
                END IF
                sn = 1.d0
                ssn = 0.d0
                DO  k=1,first-1
                   fac = 1.d0/( x(j) - x(k) )
                   sn = sn*fac*( y(i) - x(k) )
                   ssn = ssn + 1.d0/(y(i) - x(k))
                END DO
                DO  k=first+1,second-1
                   fac = 1.d0/( x(j) - x(k) )
                   sn = sn*fac*( y(i) - x(k) )
                   ssn = ssn + 1.d0/( y(i) - x(k) )
                END DO
                DO  k=second+1,nx
                   fac = 1.d0/( x(j) - x(k) )
                   sn = sn*fac*( y(i) - x(k) )
                   ssn = ssn + 1.d0/( y(i) - x(k) )
                END DO
                dp(i,j) = sn/( x(j) - x(zerfac) )
                ddp(i,j) = 2.d0*ssn*dp(i,j)
             END IF
          END IF
       END DO

    END DO

  END SUBROUTINE lgngr
END MODULE helpers

! ------------------------------------------------------------------------
subroutine DVR_reg(nfun,xbounds,pt,wt,df,eddf,ke_mat)
  ! ------------------------------------------------------------------------
  use kinddefs
  use helpers
  implicit NONE
  !------External subroutines:
  !--------"gaussq", "lgngr"------------------

  integer, INTENT(IN)   :: nfun
  real(doub_prec), dimension(1:2), INTENT(IN)  :: xbounds
  real(doub_prec), dimension(1:nfun), INTENT(OUT) :: pt 
  real(doub_prec), dimension(1:nfun), INTENT(OUT) :: wt    
  real(doub_prec), dimension(1:nfun,1:nfun), INTENT(OUT) :: ke_mat, df, eddf

  integer :: i,j,k
  real(doub_prec), dimension(1:2) :: endpoints
  real(doub_prec) :: xmin, xmax
  real(doub_prec), dimension(1:nfun)         :: scratch  
  real(doub_prec), dimension(1:nfun)         :: gauss_weights
  real(doub_prec), dimension(1:nfun)         :: roots
  real(doub_prec), dimension(1:nfun,1:nfun)  :: f, ddf
  real(doub_prec) :: A,B

  !--------------------------------------------------------------------
  ! Endpoint of the interval for the region
  !--------------------------------------------------------------------

  !write(0,*) 'DVR_reg called with:'
  !write(0,*) 'nfun = ', nfun
  !write(0,*) 'xbounds = ', xbounds

  xmax=xbounds(2)
  xmin=xbounds(1)


  !--------------------------------------------------------------------
  ! Standard interval for Legendre quadrature
  !--------------------------------------------------------------------

  endpoints(1)=-1.d0
  endpoints(2)=1.d0


  !--------------------------------------------------------------------
  ! Find zeros of nfun'th order Legendre quadrature (Gauss-Lobatto) 
  !--------------------------------------------------------------------

  call gaussq('legendre',nfun,0.d0,0.d0,2,endpoints,scratch,roots,gauss_weights)


  !--------------------------------------------------------------------
  ! Set up rescaled position grid
  !--------------------------------------------------------------------

  A=dabs(xmax-xmin)/2.d0
  B=(xmax+xmin)/2.d0
  do k=1,nfun
     pt(k)=A*roots(k)+B
     wt(k)=A*gauss_weights(k) 
  end do

  !--------------------------------------------------------------------
  ! Generate Lagrange interpolating polynomials and their first 
  ! and second derivatives
  !--------------------------------------------------------------------

  call lgngr(f,df,ddf,pt,pt,nfun,nfun,'all')


  !--------------------------------------------------------------------
  !--------------------------------------------------------------------
  ! Set up kinetic energy matrix (not normalized!)
  !--------------------------------------------------------------------
  !--------------------------------------------------------------------

  !-------------------------j=function index; i=point index----------------

  ke_mat(:,:)=0.d0
  eddf(:,:)=0.d0

  do j=1,nfun
     !******************adding the bloch contributions*******************
     ke_mat(1,j)=f(1,1)*df(1,j)
     ke_mat(nfun,j)=-f(nfun,nfun)*df(nfun,j)

     !      eddf(1,j)=f(1,1)*df(1,j)
     !      eddf(nfun,j)=-f(nfun,nfun)*df(nfun,j)
     !*****************
     do i=1,nfun
        ke_mat(i,j)=ke_mat(i,j)+ddf(i,j)*wt(i)
        eddf(i,j)=eddf(i,j)+ddf(i,j) 
     end do

  end do

  !--------------------------------------------------------------------
  ! Factor of -1/2
  !--------------------------------------------------------------------
  do i=1, nfun
     do j=1, nfun
        ke_mat(i,j)=-0.5d0*ke_mat(i,j)
     end do
  end do
end subroutine DVR_reg
