import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from upsetplot import UpSet
from upsetplot import from_indicators
import warnings
import seaborn as sns
import numpy as np
import torch
from typing import Tuple, Optional
from composition import centroids_to_kde, generate_grid

warnings.filterwarnings("ignore", category=DeprecationWarning)

def plot_image(image_path, ax, third_lines=True, width=1):
    """
    Plots an image on the given axes with an option to overlay third lines.
    """
    img = mpimg.imread(image_path)  # Load image as a NumPy array
    ax.imshow(img)
    if third_lines:
        plot_third_lines(ax, width=width)
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

def plot_centroid_evolution(image, centroids, step_interval = 10, cols = 5, plot_centroids=True, plot_saliency=True, sigma=0.05, alpha=0.4):
    """
    Plots a grid of subplots showing the evolution of centroids over time.
    """
    img_np = np.array(image)
    H, W = img_np.shape[:2]

    T, L, B, _ = centroids.shape
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(L)]

    steps = np.arange(step_interval, T + 1, step_interval)
    rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D

    t_start = 0
    for idx, step in enumerate(steps):
        for row in range(rows):
            ax = axes[row, idx]
            ax.imshow(img_np, extent=[0, img_np.shape[1], 0, img_np.shape[0]], alpha=1-alpha)
            ax.axis('off')

            if idx == 0:
              row_label = "Unconditional" if row == 0 else "Conditional"
              ax.set_ylabel(f"{row_label} attention")


            t_step = steps[idx]
            if row < 2:
                interval_centroids = centroids[t_step-1, :, row, :]
            else:
                interval_centroids = centroids[t_step-1, :, :, :]

            if plot_centroids:
                for layer in range(L):
                    scaled_centroids =  interval_centroids[layer].reshape(-1, 2) * torch.Tensor([W, H]) + torch.Tensor([W / 2, H / 2])
                    ax.scatter(scaled_centroids[:, 0],  scaled_centroids[:, 1], color=colors[layer], marker='*', s=20, linewidths=2, label=keys[layer])

            if plot_saliency:
                kde_centroids = interval_centroids.reshape(-1, 2)
                grid = generate_grid(H // 8, W //8, centered=True, grid_aspect="scaled")
                kde_map = centroids_to_kde(kde_centroids.unsqueeze(0), grid, sigma=sigma)[0]  # shape [H, W]
                print(kde_map.min(), kde_map.max())
                ax.imshow(kde_map.numpy(), extent=[0, img_np.shape[1], 0, img_np.shape[0]], origin='lower', cmap='jet', alpha=alpha)

            if row == 0:
                ax.set_title(f"T = {t_step}")


    row_labels = ["Unconditional", "Conditional"]
    for row_idx, label in enumerate(row_labels):
        fig.text(
            -0.01,                          # x position (very left)
            1 - (row_idx + 0.5) / rows,   # y position centered in row
            label,
            va='center',
            ha='right',
            fontsize=12,
        )

    plt.subplots_adjust(right=0.85)  # adjust to make room
    plt.show()