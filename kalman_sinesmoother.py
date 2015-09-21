#!/usr/bin/env python

'''
kalman_sinesmoother.py - OpenCV noisy sinewave smoothing demo using 1D Kalman filter 

Copyright (C) 2015 Simon D. Levy

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
This code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this code. If not, see <http://www.gnu.org/licenses/>.
'''

NOISEMAG = .1

from pylab import *
from numpy.random import rand

if __name__ == '__main__':

    t = arange(0.0, 2.0, 0.01)
    noise = NOISEMAG * (2 * rand(len(t)) - 1)
    s = sin(2*pi*t)
    s_noisy = s + noise
    s_filtered = zeros(t.shape)
    plot(t, s)
    plot(t, s_noisy)
    plot(t, s_filtered)

    xlabel('time (s)')
    ylabel('voltage (mV)')
    title('1D Kalman Filtering Example')
    legend(['Original', 'Noisy', 'Filtered'])
    grid(True)
    show()
