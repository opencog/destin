import matplotlib.pyplot as plt

"""
Typical usage, 
Draw a single line graph with two points:
    
charting.update([10])
charting.draw()

charting.update([4.5])
charting.draw()

Draw a chart with two line graphs:
    
charting.update([1,5])
charting.draw()

charting.update([12,45])
charting.draw()    

"""
#plt.ion()

axesis = []
data = []
the_plot = None
def update(values):
    """
        update([x-asxis, y-axis,... n-axis])
    """
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
   
def savefig(fname):
    """
        save the figure to file
    """
    plt.savefig(fname)
