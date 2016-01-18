# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:51:46 2016

@author: NARAYSR
"""
from KalmanPy import kalman
import numpy as np
import random
import matplotlib.pyplot as plt

class Voltmeter:
  def __init__(self,_truevoltage,_noiselevel):
    self.truevoltage = _truevoltage
    self.noiselevel = _noiselevel
  def GetVoltage(self):
    return self.truevoltage
  def GetVoltageWithNoise(self):
    return random.gauss(self.GetVoltage(),self.noiselevel)
    
A = np.matrix([1])
B = np.matrix([0])
H = np.matrix([1])

intq = np.matrix([0.00001])
intr = np.matrix([0.1])
intx = np.matrix([9])
intp = np.matrix([1])
uk = np.zeros(100)

voltage = []
voltmeter = Voltmeter(10.25,0.25)
for i in range(100):
    voltage.append(voltmeter.GetVoltageWithNoise())

timestmp = np.arange(0, 10, 0.1)

kalFil = kalman.kalmanFilter(timestmp, voltage)
est, err = kalFil.filter1D(A,B,H, intx, intp, intq, intr, uk)

plt.plot(timestmp,voltage,'b',timestmp,est,'g')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Voltage Measurement with Kalman Filter')
plt.legend(('measured','true voltage','kalman'))
plt.show()