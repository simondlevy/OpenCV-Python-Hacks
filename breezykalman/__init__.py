'''
breezykalman - Easy Kalman filter using OpenCV

Based on http://jayrambhia.wordpress.com/2012/07/26/kalman-filter/

Copyright (C) 2014 Simon D. Levy

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
This code is distributed in the hope that it will be useful,

MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this code. If not, see <http://www.gnu.org/licenses/>.
'''

import cv

class BreezyKalman(object):
    '''
    A class for easy Kalman filtering
    '''

    def __init__(self, dims, processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1):
        '''
        Constructs a new BreezyKalman object with specified number of input/output dimensions
        For explanation of the error covariances see
        http://en.wikipedia.org/wiki/Kalman_filter
        '''

        self.dims = dims

        self.kalman = cv.CreateKalman(dims*2, dims, 0)
        self.kalman_state = cv.CreateMat(dims*2, 1, cv.CV_32FC1)
        self.kalman_process_noise = cv.CreateMat(dims*2, 1, cv.CV_32FC1)
        self.kalman_measurement = cv.CreateMat(dims, 1, cv.CV_32FC1)

        for j in range(dims*2):
            for k in range(dims*2):
                self.kalman.transition_matrix[j,k] = 0
            self.kalman.transition_matrix[j,j] = 1

        cv.SetIdentity(self.kalman.measurement_matrix)

        cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(processNoiseCovariance))
        cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))
        cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(errorCovariancePost))

        self.predicted = None
        self.corrected = None

    def update(self, obs):
        '''
        Updates the filter with a new observation
        '''

        for k in range(self.dims):
            self.kalman_measurement[k,0] = obs[k]

        self.predicted = cv.KalmanPredict(self.kalman)
        self.corrected = cv.KalmanCorrect(self.kalman, self.kalman_measurement)

    def getEstimate(self):
        '''
        Returns the current X,Y estimate.
        '''

        return [self.corrected[k,0] for k in range(self.dims)]

    def getPrediction(self):
        '''
        Returns the current X,Y prediction.
        '''

        return [self.predicted[k,0] for k in range(self.dims)]
