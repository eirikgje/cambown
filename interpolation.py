import numpy as np

def cubic_spline(x, y):
    """ Natural cubic spline based on the points x and y. 
    
    Returns the second derivatives.
    
    """

    n = len(x)
    c = np.zeros(n)
    r = np.zeros(n)
    a = np.zeros(n)
    b = np.zeros(n)
    c[:n-1] = x[1:] - x[:n-1]
    r[:n-1] = 6 * (y[1:] - y[:n-1]) / c[:n-1]
    r[1:n-1] = r[1:n-1] - r[:n-2]
    a[1:n-1] = c[:n-2]
    b[1:n-1] = 2 * (c[1:n-1] + a[1:n-1])
    b[0] = 1.0
    b[n-1] = 1.0
    r[0] = 0
    c[0] = 0
    r[n-1] = 0
    a[n-1] = 0

    gam = np.zeros(n)
    bet = b[0]
    y2 = np.zeros(n)
    y2[0] = r[0] / bet
    for i in range(1, n):
        gam[i] = c[i-1] / bet
        bet = b[i] - a[i] * gam[i]
        y2[i] = (r[i] - a[i] * y2[i-1]) / bet
    for i in range(n-2, -1, -1):
        y2[i] -= gam[i+1] * y2[i+1]

    return y2

def splint(xa, ya, y2a, x):
    """ Splints the spline defined by xa, ya, and y2a at point(s) x. """

#    print xa.shape
#    print x
    khi = np.searchsorted(xa, x)
    klo = khi - 1
#    print khi, klo
    h = xa[khi] - xa[klo]
#    sys.exit()
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h
    return a * ya[klo] + b * ya[khi] + ((a ** 3 - a) * y2a[klo] + (b**3 - b) * y2a[khi]) * (h ** 2) / 6.0

def splint_deriv(xa, ya, y2a, x):
    """ Splints the derivative of the function defined by the spline defined by xa, ya, y2a, at x """

    khi = np.searchsorted(xa, x)
    klo = khi - 1
    h = xa[khi] - xa[klo]
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h

    return (ya[khi] - ya[klo]) / h - (3 * a ** 2 - 1) / 6 * h * y2a[klo] + (3 * b**2 - 1) / 6 * h * y2a[khi]

