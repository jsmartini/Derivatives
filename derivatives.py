from scipy.stats import norm
import numpy as np
from numpy import log, sqrt, exp
import numpy as np
from matplotlib import pyplot as plt
from typing import *
from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta as tdelta
from enum import Enum

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


# enums to describe the trade details

class OptionType(Enum):
    Call = 0
    Put = 1
    Forward = 2

class ExerciseType(Enum):
    # use a wrapper function for each contract to return and execution flag in the future for monte carlo
    asian = 0
    european = 1
    american = 2

class OptionSide(Enum):
    Short = 0
    Long = 1

class TimeFrequency(Enum):
    Day = 0
    Week = 1
    Month = 2
    Year = 3

def time2float(td: tdelta, config: TimeFrequency):
    total = td.seconds
    match config:
        case TimeFrequency.Day: return td.seconds#total / tdelta(days=1).seconds
        case TimeFrequency.Week: return td.weeks#total / tdelta(weeks=1).seconds
        case TimeFrequency.Month: return td.months#total / tdelta(months=1).seconds
        case TimeFrequency.Year: return td.years#total / tdelta(years=1).seconds


"""
    OptionStrategy:
    - Constructor:
        - Derivatives, list of partial functions to evaluate the entire position
    
"""

@dataclass
class Leg:                      # describes a subcontract of a strategy
    Side : OptionSide
    Type : OptionType
    Strike:  float              # strike price for the underlying
    Expiry:  datetime  # expiration date in years
    Riskfree: float             # risk free rate   
    Dividend: float             # continuous dividend rate
    Volatility: float           # standard deviation 
    Spot: float
    Tfreq: TimeFrequency 
    Quantity: int

    def __str__(self) -> str:
        return f"{self.Side.name} {self.Type.name} @ strike {self.Strike} expiring {self.Expiry}"

def evaluationFunction(Spots, contract: Leg):
    match contract.Type:
        case OptionType.Call:
            # black scholes
            return LCall(Spots, contract.Strike) if contract.Side == OptionSide.Long else SCall(Spots, contract.Strike)
        case OptionType.Put:
            # black scholes
            return LPut(Spots, contract.Strike) if contract.Side == OptionSide.Long else SPut(Spots, contract.Strike)
        case OptionType.Forward:
            # fair value for forward contracts
            return LForward(Spots, contract.Strike) if contract.Side == OptionSide.Long else SForward(Spots, contract.Strike)

def valuationFunction(duration: float, contract: Leg):
    # extendable valutation function
    bs_args = [contract.Spot, contract.Strike, duration, contract.Riskfree, contract.Dividend, contract.Volatility]
    match contract.Type:
        case OptionType.Call:
            # black scholes
            return bs_call(*bs_args)* (-1 if contract.Side == OptionSide.Short else 1)
        case OptionType.Put:
            # black scholes
            return bs_put(*bs_args) *(-1 if contract.Side == OptionSide.Short else 1)
        case OptionType.Forward:
            # fair value for forward contracts
            return contract.Spot * np.exp(duration*(contract.Riskfree-contract.Dividend))

class OptionStrategy(object):

    def __init__(self,current_date: datetime,  derivatives: List[Leg]):
        self.current_date = current_date
        spot_max = int(derivatives[0].Strike *3)
        self.Spots = np.linspace(0,spot_max, 100*spot_max)
        self.Derivatives = derivatives
        self.Parts  = List[Dict]
        self.payoff = np.zeros(self.Spots.shape)
        self.valuation = 0

    def eval(self):
        self.Parts = list()    # clear old evaluations
        for derivative in self.Derivatives:     
            time = time2float(tdelta(derivative.Expiry,self.current_date), derivative.Tfreq)   # convert to time float
            # easy unpack in the "bsvalue" key
            self.Parts.append(
                {
                "Leg": derivative,
                "payoff": evaluationFunction(self.Spots, derivative),
                "value": valuationFunction(time, derivative)
                }
            )
        self.payoff = np.sum([component["payoff"]*component["Leg"].Quantity for component in self.Parts], axis=0)
        self.valuation = np.sum([component["value"]*component["Leg"].Quantity*(-1 if component["Leg"].Side == OptionSide.Long else 1) \
             for component in self.Parts])

    def changeFieldAndReEval(self, field, value):
        for idx, obj in enumerate(self.Derivatives):
            try:
                setattr(obj, field, value)
            except BaseException as e:
                print(e)
                continue
            self.Derivatives[idx] = obj
        self.eval()
    
    def reportData(self):
        return {
            "parts": self.Parts,
            "payoff": self.payoff,
            "valuation": self.valuation
        }

    def __str__(self):
        return "\n".join(["Strategy Composition"] + [
            f"{p['Leg']} ${round(p['value'],2)}" for p in self.Parts
        ])

    def plot(self):
        fig, ax = plt.subplots()
        legends = ["OptionStrategy", "min", "max"]
        ax.plot(self.Spots, self.payoff)
        ax.plot(self.Spots[self.payoff == np.min(self.payoff)], self.payoff[self.payoff == np.min(self.payoff)])
        ax.plot(self.Spots[self.payoff == np.max(self.payoff)], self.payoff[self.payoff == np.max(self.payoff)])
        for part in self.Parts:
            ax.plot(self.Spots, part["payoff"])
            legends.append(str(part["Leg"]))
        fig.legend(legends)
        plt.show()



        
    
    
        


















