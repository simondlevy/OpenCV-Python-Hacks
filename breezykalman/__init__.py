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

    def __init__(self, n, m, processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1):
        '''
        Constructs a new BreezyKalman object with specified number of state and measurement dimensions.
        For explanation of the error covariances see
        http://en.wikipedia.org/wiki/Kalman_filter
        '''

        self.n = n
        self.m = m

        self.kalman = cv.CreateKalman(n, m, 0)
        self.kalman_state = cv.CreateMat(n, 1, cv.CV_32FC1)
        self.kalman_process_noise = cv.CreateMat(n, 1, cv.CV_32FC1)
        self.kalman_measurement = cv.CreateMat(m, 1, cv.CV_32FC1)

        for j in range(n):
            for k in range(n):
                self.kalman.transition_matrix[j,k] = 0
            self.kalman.transition_matrix[j,j] = 1

        cv.SetIdentity(self.kalman.measurement_matrix)

        cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(processNoiseCovariance))
        cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))
        cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(errorCovariancePost))

    def step(self, obs):
        '''
        Runs one pass of the filter prediction/update, returning a new state estimate.
        '''

        for k in range(self.m):
            self.kalman_measurement[k,0] = obs[k]

        cv.KalmanPredict(self.kalman)

        estimated = cv.KalmanCorrect(self.kalman, self.kalman_measurement)

        return tuple([estimated[k,0] for k in range(self.m)])
