#!/usr/bin/python

import numpy as np
from scipy.optimize import least_squares


class Fitter(object):
    """
    Orthogonal distance regression (ODR) fitter object.
    """

    def __init__(self,):
        pass

    def inv(self, mat):
        """
        Tries to invert a matrix the normal way, otherwise
        uses pinv.

        """
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError as err:
            # print("here")
            if 'Singular matrix' in str(err):
                # print("here2")
                print("Matrix was singular, using pinv.")
                return np.linalg.pinv(mat)
            else:
                raise ValueError("Couldn't invert matrix")

    def Model(self, function):
        """
        Set the model to fit to.

        Parameters
        ----------
        function : the fit function used for the fit.  Must return
                   an array the same shape as ydata.

        """
        self.model = function

    def Beta0(self, init_beta):
        """
        Set the initial parameters.

        Parameters
        ----------
        init_beta : (array-like) initial guess for the parameters.

        """
        self.beta0 = np.array(init_beta)
        self.bs = len(init_beta)

    def Data(self, xdata, ydata, xevaldata=None, sigmax=None, sigmay=None,
             covx=None, covy=None):
        """
        Input the x and y data, as well as the errors if they exist.

        Parameters
        ----------
        xdata     : The x-data used in fitting.  Doesn't have to have the same
                    length as ydata.
        ydata     : The y-data used in fitting.
        xevaldata : (Optional, default=None) The xdata to use in evaluating
                    the function.
                    If not None, the additional x-data to be evaluated are
                    assumed to be inserted on the left side of xdata,
                    that is at index zero.
        sigmax    : (Optional, default=None) The errors for the x data.
        sigmay    : (Optional, default=None) The errors for the y data.
        covx      : (Optional, default=None) The covariance matrix for
                    the x-data.  The square-root of the diagonal of this
                    matrix are the x-errors.
        covy      : (Optional, default=None) The covariance matrix for
                    the y-data.  The square-root of the diagonal of this
                    matrix are the y-errors.

        """
        # if (len(xdata) != len(ydata)):
        #     raise ValueError("x and y data must have same shape")
        self.y = ydata
        self.x = xdata
        if (xevaldata is not None):
            self.xed = xevaldata
        else:
            self.xed = self.x
        self.xs = len(xdata)
        self.ys = len(ydata)
        self.xeds = len(self.xed)
        self.delta = np.zeros(shape=(self.xs,))
        if (sigmax is not None) and (sigmay is not None):
            # sx and sy
            assert (covx is None) and (covy is None)
            Omega = np.block([[np.diag(1./sigmay**2), np.zeros((self.ys, self.xs))],
                              [np.zeros((self.xs, self.ys)), np.diag(1./sigmax**2)]])
        elif (sigmax is not None) and (sigmay is None) and (covx is None) and (covy is None):
            # sx 
            Omega = np.block([[np.eye(self.ys), np.zeros((self.ys, self.xs))],
                              [np.zeros((self.xs, self.ys)), np.diag(1./sigmax**2)]])
        elif (sigmax is None) and (sigmay is not None) and (covx is None) and (covy is None):
            # sy
            Omega = np.block([[np.diag(1./sigmay**2), np.zeros((self.ys, self.xs))],
                              [np.zeros((self.xs, self.ys)), np.eye(self.xs)]])
        elif (covx is not None) and (covy is not None):
            # cx and cy
            assert (sigmax is None) and (sigmay is None)
            Omega = np.block([[self.inv(covy), np.zeros((self.ys, self.xs))],
                              [np.zeros((self.xs, self.ys)), self.inv(covx)]])
        elif (covx is None) and (covy is not None) and (sigmax is None) and (sigmay is None):
            # cy
            Omega = np.block([[self.inv(covy), np.zeros((self.ys, self.xs))],
                              [np.zeros((self.xs, self.ys)), np.eye(self.xs)]])
        elif (covx is not None) and (covy is None) and (sigmax is None) and (sigmay is None):
            # cx
            Omega = np.block([[np.eye(self.ys), np.zeros((self.ys, self.xs))],
                              [np.zeros((self.xs, self.ys)), self.inv(covx)]])
        elif (sigmax is not None) and (covy is not None):
            # sx and cy
            assert (covx is None) and (sigmay is None)
            Omega = np.block([[self.inv(covy), np.zeros((self.ys, self.xs))],
                              [np.zeros((self.xs, self.ys)), np.diag(1./sigmax**2)]])
        elif (sigmay is not None) and (covx is not None):
            # cx and sy
            assert (covy is None) and (sigmax is None)
            Omega = np.block([[np.diag(1./sigmay**2), np.zeros((self.ys, self.xs))],
                              [np.zeros((self.xs, self.ys)), self.inv(covx)]])
        else:
            # print("seven")
            Omega = np.eye(self.ys+self.xs)

        self.L = np.linalg.cholesky(Omega)

    def Residuals(self, z):
        """
        Calculates the residuals for ODR.

        Parameters
        ----------
        z : array with initial guess for parameters, and
            inital values for x variations.

        Returns
        -------
        res : the residuals whose square is used in minimization.

        """
        para = z[:self.bs]
        arguments = z[self.bs:]

        rx = arguments
        rxeval = np.hstack((np.zeros((self.xeds-self.xs,)), rx))
        ry = (self.model(self.xed + rxeval, *para) - self.y)
        r = np.hstack((ry, rx))

        return np.dot(r, self.L)

    def Run(self,):
        """
        Runs the fitter.

        Notes
        -----
        params : The final values of the fit parameters.
        chi2   : The chi squared per degree of freedom.
        pcov   : The errors on the fit parameters.

        """
        whole = np.append(self.beta0, self.delta)
        self.out = least_squares(self.Residuals, whole, method='lm')
        # self.out = minimize(self.Residuals, whole)
        assert self.out.success

        self.chi2 = np.sum(self.out.fun**2) / float(len(self.y) - len(self.beta0))

        self.params = self.out.x[:self.bs]

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = np.linalg.svd(self.out.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(self.out.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        self.pcov = np.dot(VT.T / s**2, VT)[:self.bs, :self.bs]
