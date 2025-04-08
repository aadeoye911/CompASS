import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def plot_image(image_path, ax, third_lines=True):
    """
    Plots an image on the given axes with an option to overlay third lines.
    """
    img = mpimg.imread(image_path)  # Load image as a NumPy array
    ax.imshow(img)
    if third_lines:
        plot_third_lines

def plot_third_lines(ax, color='red', style='--'):
    ax.axvline(x=ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 3, color=color, linestyle=style)
    ax.axhline(y=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 3, color=color, linestyle=style)
    ax.axvline(x=ax.get_xlim()[0] + 2 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / 3, color=color, linestyle=style)
    ax.axhline(y=ax.get_ylim()[0] + 2 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 3, color=color, linestyle=style)

def plot_colorbar(img, fig, ax, location="bottom", orientation="horizontal"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="7%", pad="10%")
    fig.colorbar(img, cax=cax, orientation="horizontal")