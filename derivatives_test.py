import pytest
from derivatives import *
from derivative_plot import plot_derivative
import datetime

def ABCBullSpreadProblem():
    price = 46
    volatility = .4
    interest = .04
    dividend = 0
    current = datetime.datetime(year=1, month =1, day =1)
    derivatives = [
        Leg(
            Side=OptionSide.Long,
            Type=OptionType.Call,
            Strike=45,
            Expiry=datetime.datetime(year=2,month =1, day =1),
            Riskfree=interest,
            Dividend=dividend,
            Volatility=volatility,
            Spot=price,
            Tfreq=TimeFrequency.Year,
            Quantity=1
        ),
        Leg(
            Side=OptionSide.Short,
            Type=OptionType.Call,
            Strike=55,
            Expiry=datetime.datetime(year=2, month =1, day =1),
            Riskfree=interest,
            Dividend=dividend,
            Volatility=volatility,
            Spot=price,
            Tfreq=TimeFrequency.Year,
            Quantity=1
        )
    ]
    bullspread = OptionStrategy(
        current_date=current,
        derivatives=derivatives
    )
    bullspread.eval()
    print(bullspread)
    assert type(bullspread.payoff) == np.ndarray
    assert type(bullspread.Spots) == np.ndarray
    bullspread.plot()
    return bullspread.reportData()


if __name__ == "__main__":
    ABCBullSpreadProblem()