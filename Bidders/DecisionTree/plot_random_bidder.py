import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# plots for random biidding data

class Chunk:
    
    chunk_ind = 0
    lower_bound = 0.0
    upper_bound = 1.0
    cpc = 0.0
    ctr = 0.0
    bids_won = 0
    total_clicks = 0
    money_spent = 0.0

def load_data():
    """
    Read data from the text file that was produced by the random bidder
    """
    tests = []
    with open("tuned_80_rounds.txt") as f:
        chunk = None
        for l in f.readlines():
            if "Tuning finished" in l:
                break
            elif "On test " in l:
                chunk = Chunk()
                ind = int(l.split()[-1])
                chunk.chunk_ind = ind
            elif "using bounds" in l:
                bounds = l.split()
                lb = float(bounds[2])
                ub = float(bounds[4])
                chunk.lower_bound = lb
                chunk.upper_bound = ub
            elif "total clicks" in l:
                c = l.split()
                c = int(c[-1])
                chunk.total_clicks = c
            elif "total paid" in l:
                chunk.money_spent = float(l.split()[-1])
            elif "bids won" in l:
                chunk.bids_won = int(l.split()[-1])
            elif "CTR" in l:
                chunk.ctr = float(l.split()[-1])
            elif "CPC" in l:
                chunk.cpc = float(l.split()[-1])
                tests.append(chunk)
                chunk = None
    return tests
            

def plot_data(data):
    plotting = []
    for chunk in data:
        plotting.append((chunk.lower_bound, chunk.upper_bound, chunk.cpc))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(len(plotting))
    ax.scatter(*zip(*plotting), c='r', marker='o')
    plt.title("Click through Rate for each bound")
    ax.set_xlabel("Lower Bound")
    ax.set_ylabel("Upper Bound")
    ax.set_zlabel("Click through Rate")
    plt.show()

    


d = load_data()
plot_data(d)