# -*- coding: utf-8 -*-
"""
This module provides Kalman filtering class.
Kalman filter has wide application domain. This class will begin to provide
basic function as 1d filter and future features depending on demand.

Created on Tue Dec 22 18:00:00 2015

@author: Srikanth Narayanan
"""

import numpy as np

__author__= "Srikanth Narayanan"
__version__= "1.0.0"
__email__="srikanth.n.narayanan@gmail.com"

class kalmanFilter(object):
    '''
    A class contains various features of a Kalman filter.
    '''
    def __init__(self, time, signal):
        '''
        Constructor to initialise kalman filter object. Needs to basic input time
        and signal vectors.
        
        :param time: a list or numpy vector of time stamp
        :param signal: a list or numpy vector of signal
        '''
        self.userTime = np.array(time)
        self.userSignal = np.array(signal)
        self.matType = np.matrixlib.defmatrix.matrix
    
    def _crucify(self):
        '''
        Method to crucify the object while creation when it does not meet the 
        requirement.
        '''
        del self
    
    def filter1D(self, A, B, H, initX, initP, initQ, initR, Uk):
        '''
        This method implements barebone 1d kalaman filter. The efficiency 
        depends on the user input of key kalman parameters.
        
        :param A: State transition matrix as type numpy matrix
        :param B: Control matrix as type numpy matrix
        :param H: Observation matrix as type numpy matrix
        :param initX: Initial state estimate as type numpy matrix
        :param initP: Initial Covariance estimate as type numpy matrix
        :param initQ: Initial process error estimate as type numpy matrix
        :param initR: Initial measurement error estimate as type numpy matrix
        :param Uk: Control Vector
        '''
        self.A = A
        self.B = B
        self.H = H
        self.curStateEst = initX
        self.curProbEst = initP
        self.Q = initQ
        self.R = initR
        self.Uk = Uk
        self.outStateEst = []
        self.outProbEst = []
        
        #check if the vector of same length
        if self.Uk.shape == self.userSignal.shape:
            for i in range(len(self.userSignal)):
                # Predict
                self._predictKal(np.matrix(Uk[i]))
                # Observe
                self._observeKal(np.matrix(self.userSignal[i]))
                # Update
                self._updateKal()
                self.outStateEst.append(self.curStateEst.A1)
                self.outProbEst.append(self.curProbEst.A1)
            
            #convert to np array
            self.outProbEst = np.array(self.outProbEst)
            self.outStateEst = np.array(self.outStateEst)
            
            return self.outStateEst, self.outProbEst
        else:
            print "Error: Both time and signal vector should be of the same length"
    
    def _predictKal(self, stepUk):
        '''
        Helper method for predict phase of the filter
        '''
        self.predStateEstimate = (self.A * self.curStateEst) + (self.B * stepUk)
        self.predErrorEstimate = (self.A * self.curProbEst * self.A.T) + self.Q 
    
    def _observeKal(self, measuredValue):
        '''
        Helper method for error observation phase of the filter
        '''
        self.observePrediction = measuredValue - (self.H * self.predStateEstimate)
        self.observeCovariance = (self.H * self.predErrorEstimate * self.H.T) + self.R
    
    def _updateKal(self):
        '''
        Helper method for update phase of the filter
        '''
        self.kalmanGain = self.predErrorEstimate * self.H.T * np.linalg.inv(self.observeCovariance)
        self.curStateEst = self.predStateEstimate + self.kalmanGain * self.observePrediction
        
        # Create Identity Matrix of size of current Probability estimate
        Imat = np.eye(self.curProbEst.shape[0])
        
        # Update Probability estimate        
        self.curProbEst = (Imat - self.kalmanGain * self.H) * self.predErrorEstimate
    
    def plot1D(self):
        '''
        This method plots the 1d kalman filtered object in comparison to the 
        given user signal. It also plots the error covariance.
        '''
        
    