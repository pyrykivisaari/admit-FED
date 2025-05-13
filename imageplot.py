
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plotIt(filename):
    image = mpimg.imread(filename)
    fig1, ax1 = plt.subplots()
    ax1.imshow(image)
    ax1.set_axis_off()
    plt.show()

