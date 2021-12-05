import numpy as np
import time
import sched


class PID:

    def __init__(self, kp=1, ki=0, kd=0, sampleTime=0.01):
        """
        :param kp: Proportional Gain
        :param ki: Integral Gain
        :param kd: Derivative Gain
        :param sampleTime: Sampling Time
        """
        # Gains + Low-Pass Filter
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau = 0.02

        # TIme
        self.sampleTime = sampleTime

        # Setpoint + Output + Measurements
        self.setPoint = None
        self.output = None
        self.measurement = np.array([0, 0, 0], dtype=float)
        self.prevMeasurement = np.array([0, 0, 0], dtype=float)

        # Error terms
        self.error = np.array([0, 0, 0], dtype=float)
        self.prevError = np.array([0, 0, 0], dtype=float)

        # Values of each term
        self.pTerm = np.array([0, 0, 0], dtype=float)
        self.iTerm = np.array([0, 0, 0], dtype=float)
        self.dTerm = np.array([0, 0, 0], dtype=float)

        # Integral Anti-Windup + Output Limits
        self.intMax = 15
        self.intMin = -15
        self.outMin = -20
        self.outMax = 20

    def calculate(self):
        """PID Controller in Z Domain"""
        # Calculating Error
        self.error = self.setPoint - self.measurement

        # Proportional Term Calculation
        self.pTerm = self.kp * self.error

        # Integral Term Calculation
        self.iTerm += (self.error + self.prevError) * self.sampleTime * self.ki * 0.5

        # Integral Anti-Windup
        self.iTerm = np.where(self.iTerm > self.intMax, self.intMax, self.iTerm)
        self.iTerm = np.where(self.iTerm < self.intMin, self.intMin, self.iTerm)

        # Derivative Term Calculation
        self.dTerm = -(2.0 * self.kd * (self.measurement - self.prevMeasurement)
                       + (2.0 * self.tau - self.sampleTime) * self.kd) / (2.0 * self.tau + self.sampleTime)

        # Output Limit
        self.output = np.where(self.output > self.outMax, self.outMax, self.output)
        self.output = np.where(self.output < self.outMin, self.outMin, self.output)

        # Setting Previous Error and Measurement
        self.prevError = self.error
        self.prevMeasurement = self.measurement

    def getSetPoint(self, vector: np.array):
        self.setPoint = vector

    def getMeasurement(self, vector: np.array):
        self.measurement = vector


if __name__ == '__main__':
    pid = PID()
