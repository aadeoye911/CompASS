import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from upsetplot import UpSet
from upsetplot import from_indicators
import warnings
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

def plot_image(image_path, ax, third_lines=True):
    """
    Plots an image on the given axes with an option to overlay third lines.
    """
    img = mpimg.imread(image_path)  # Load image as a NumPy array
    ax.imshow(img)
    if third_lines:
        plot_third_lines(ax)
    ax.axis("off")

def plot_third_lines(ax, color='red', style='--', width=0.2):
    ax.axvline(x=ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 3, color=color, linestyle=style, linewidth=width)
    ax.axhline(y=ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 3, color=color, linestyle=style, linewidth=width)
    ax.axvline(x=ax.get_xlim()[0] + 2 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / 3, color=color, linestyle=style, linewidth=width)
    ax.axhline(y=ax.get_ylim()[0] + 2 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 3, color=color, linestyle=style, linewidth=width)

def plot_colorbar(img, fig, ax, location="bottom", orientation="horizontal"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="7%", pad="10%")
    fig.colorbar(img, cax=cax, orientation="horizontal")

def plot_multilabel_distribution(df, sort_by="degree", facecolor="darkred"):
    # Ensure boolean type for indicators
    df = df.astype(bool)
    data = from_indicators(indicators=df.columns, data=df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        upset = UpSet(data, 
                    sort_by=sort_by,
                    min_degree=1,
                    sort_categories_by="input",
                    facecolor=facecolor,
                    show_counts=True)
        upset.plot()
        plt.show()

def plot_ar_distribution(labels_df):
    labels_df["aspect_ratio"] = labels_df.apply(lambda row: round(row["width"] / row["height"], 2), axis=1)
    labels_df["resized_ar"] = labels_df.apply(lambda row: round(row["resized_width"] / row["resized_height"], 2), axis=1)

    # Calculate modal values and their KDE peak heights for annotation
    resized_peaks =[labels_df["resized_ar"].mode()[0], labels_df["resized_ar"].value_counts().index[1]]

    # Plot KDEs to extract values for peaks
    plt.figure(figsize=(12, 2))
    kde_raw = sns.kdeplot(labels_df["aspect_ratio"], label="Raw Data")
    kde_resized = sns.kdeplot(labels_df["resized_ar"], label="Resized Data")

    # Extract KDE data
    x_resized, y_resized = kde_resized.get_lines()[1].get_data()

    # Draw vertical lines at modal values
    for value in resized_peaks:
        idx = np.argmin(np.abs(x_resized - value))
        y_peak = y_resized[idx]
        plt.vlines(value, 0, y_peak, color=kde_resized.get_lines()[1].get_color(), linestyle='--')

    # Adjust x-axis ticks and labels
    plt.xticks(sorted(set(list(plt.xticks()[0]) + resized_peaks)))
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# def visualize_latents(sampler):
#     T = len(sampler.decoded_images)
#     cols = min(10, T)  # Max 10 columns
#     rows = (T + cols - 1) // cols
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

#     # Plot images in grid
#     for i, ax in enumerate(axes.flatten()):
#         if i < len(sampler.decoded_images):
#             ax.imshow(sampler.decoded_images[i])
#             ax.set_title(f"$z_{{{T -i}}}$", fontsize=12)
#         ax.axis("off")

#     # Tight layout for better spacing
#     plt.tight_layout()
#     plt.show()