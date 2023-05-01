import radmc3dPy as r3d
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_r3d_image():
    try:
        im = r3d.image.readImage()  # read image.out
    except:
        print("No image.out found.")
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    r3d.image.plotImage(im, au=True, log=False, cmap=mpl.cm.inferno)
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_aspect('equal')
    plt.savefig('r3d_output.png')


if __name__ == '__main__':
    plot_r3d_image()
