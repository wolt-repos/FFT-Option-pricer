#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:49:22 2021

@author: walt
"""

import numpy as np
from scipy.stats import norm
from scipy.fft import fft
import matplotlib.pyplot as plt

class FFT_Option_pricer:
    
    def __init__(self, S0, r, sigma, T):
        
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        #self.K = K
        
        
        self.h = 0.55
        self.N = 2 ** 7
        
        self.T = T
        #self.a = 10
        self.lamda = 2 * np.pi / (self.N * self.h)
        
        
        #self.h = self.a / (self.N - 1)
        
        self.b = self.lamda * (self.N - 1) / 2
        
        self.ku = np.array([-self.b + self.lamda * u for u in np.arange(self.N)])
        
        self.K = np.exp(self.ku)
        #print(self.ku)
        
        self.alpha = 0.6
        
    def phi(self, u):
        # characteristic function
        # dS = rSdt + sigma*dW
        # E(e^iu*sT), sT=ln(ST)
        return np.exp(-0.5 * (u * self.sigma) ** 2 * self.T + 1.j * u * (np.log(self.S0) + (self.r - 0.5 * self.sigma ** 2) * self.T))
    
    def psi(self, v):
        
        return np.exp(-self.r * self.T) * self.phi(v - (self.alpha + 1) * 1.j) / \
            (self.alpha ** 2 + self.alpha - v ** 2 + 1.j * (2 * self.alpha + 1) * v)
            
    def BlackScholes_formula(self):        
        
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        c = self.S0 * norm.cdf(d1, loc=0, scale=1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2, loc=0, scale=1)
        return c
    
    def FFT_price(self):
        
        pseudo_delta_fun = lambda n: 1 if n == 0 or n == self.N - 1 else 0
        #print(delta(2 ** 10))
        x = np.array([np.exp(1.j * n * self.h * self.b) * self.psi(n * self.h) * (2 - pseudo_delta_fun(n)) for n in np.arange(self.N)])
        y =  np.exp(-self.alpha * self.ku) * self.h / (2 * np.pi) * fft(x)
        #print(y.real)
        return(y.real)
    
    
    def plot(self):
        
        plt.plot(self.K, self.BlackScholes_formula(), label = 'BS')
        plt.plot(self.K, self.FFT_price(), 'r+', label = 'FFT')
#        plt.plot(self.X[0, 2, :], label = 'd3')
        plt.legend()
        
    
ft = FFT_Option_pricer(100, 0.3, 0.2, 1.0)
ft.plot()
ft.FFT_price()