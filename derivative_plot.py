
import matplotlib.pyplot as plt
import numpy as np

def group_assert(vars: list, types):
    if type(types) == list:
        for v, t in zip(vars, types): assert v == t
    else: 
        for v in vars:                
            assert v == types


def plot_derivative(dfunc: np.array, Spots: np.array, title="profit", payprof = True):
    fig, ax = plt.subplots()
   
    ax.set_title(title)
    def zero(arry: np.array):
        closest = arry[0]
        closest_idx = 0
        for idx, val in enumerate(arry):
            if np.abs(val) < np.abs(closest):
                closest_idx = idx
                closest = val
        return [closest, closest_idx]
    val, idx = zero(dfunc)
    print(Spots[idx])
    ax.plot(Spots, dfunc)
    ax.plot(Spots[idx], val,"ro")# f"Break Even ${round(Spots[idx],2)}")
    ax.plot(Spots[dfunc == np.min(dfunc)], dfunc[dfunc == np.min(dfunc)])
    ax.plot(Spots[dfunc == np.max(dfunc)], dfunc[dfunc == np.max(dfunc)])
    ax.grid(True, which='both')

    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position('zero')
    ax.set_ylim(np.min(dfunc), max(dfunc))
    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    flag = "profit" if payprof else "payoff"
    ax.legend([
        f"Derivative {flag}", 
        f"Break Even Point: ${round(Spots[idx],2)}",
        f"Minimum {flag} ${round(np.min(dfunc),2)}",
        f"Maximum {flag} ${round(np.max(dfunc),2)}"
        ])
    ax.set_ylabel("Profit $")
    plt.show()