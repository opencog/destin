import matplotlib.pyplot as plt

#plt.ion()

axesis = []
data = []
the_plot = None
def update(values):
    for index, val in enumerate(values):
        if(index >= len(data)):
            data.append([])
        data[index].append(val)
    
    
def draw():
    if(len(data) == 0):
        return
    
    # Clear axis so it will replace the current graph
    # instead of adding a new one
    plt.cla() 
    
    for axises in data:
        plt.plot(axises)
    plt.pause(.01) # give the chart a change to draw itself
   
