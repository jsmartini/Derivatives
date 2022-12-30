from scipy.stats import norm
import numpy as np
from numpy import log, sqrt, exp
import numpy as np
from matplotlib import pyplot as plt
from typing import *
from dataclasses import dataclass
from datetime import datetime
import timedelta

"""
    Black Scholes Evaluation Functions
"""

def d1(S,K,T,r,dr,sigma):
    return(log(S/K)+(r-dr+sigma**2/2.)*T)/(sigma*sqrt(T))

def d2(S,K,T,r,dr,sigma):
    return d1(S,K,T,r,dr,sigma)-sigma*sqrt(T)

def bs_call(S,K,T,r,dr,sigma):
    return S*exp(-dr*T)*norm.cdf(d1(S,K,T,r,dr,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,dr,sigma))
  
def bs_put(S,K,T,r,dr,sigma):
    return K*exp(-r*T)-S*exp(-dr*T)+bs_call(S,K,T,r,dr,sigma)

"""
    Financial Derivatives Building Blocks
    - Long Call
    - Long Put
    - Short Call
    - Short Put
    - Long Forward
    - Short Forward
"""
LCall = lambda S, K: np.maximum(0, S-K)
LPut = lambda S, K: np.maximum(0, K-S)
SCall =  lambda S, K: np.minimum(0, K-S)
SPut =  lambda S, K: np.minimum(0, S-K)
LForward = lambda S, K: S-K
SForward = lambda S, K: K-S


"""
    OptionStrategy:
    - Constructor:
        - Derivatives, list of partial functions to evaluate the entire position
    
"""

@dataclass
class Leg:
    partial: function           # function handle
    strike:  float              # strike price for the underlying
    expiry:  int                # expiration date in years
    riskfree: float             # risk free rate   
    dividend: float             # continuous dividend rate
    volatility: float           # standard deviation 

class OptionStrategy(object):

    def __init__(self,current_date: datetime,  Derivatives: List[Leg], spot_max = 750):
        self.current_date = current_date
        self.Spots = np.linspace(0,spot_max, 100*spot_max)
        self.Derivatives = Derivatives
        self.Parts  = List[Tuple[Leg, np.array]]    # contains sum of parts
        self.payoff = np.zeros(self.Spots.shape)

    def payoff(self):
        for derivative in self.Derivatives:     self.Parts.append(tuple(derivative, derivative.function(self.Spots)))
        self.payoff = np.sum([payoff for (_, payoff) in self.Parts])
    
    def bsvalue(self):
        
        
        
    
    
        


















