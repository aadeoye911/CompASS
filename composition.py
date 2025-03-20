import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F

def minmax_normalization(attn_map):
    """ 
    Min-max normalization
    """
    min = attn_map.min()
    max = attn_map.max()
    # if max == min:
    #     return torch.zeros_like(attn_map) # Avoid division by zero

    return (attn_map - min) / (max - min)

def z_normalization(attn_map):
    """ 
    Z normalization
    """
    mean = attn_map.mean()
    std = attn_map.std()
    # if torch.isclose(std, torch.tensor(0.0), atol=1e-8)
    #     return torch.zeros_like(attn_map) # Avoid division by zero
    return (attn_map - mean) / std

def softmax_normalization(attn_map, temperature=1.0):
    """ 
    Softmax normational
    """
    attn_map = attn_map / temperature  # Optional: Adjust distribution sharpness
    attn_map = torch.nn.functional.softmax(attn_map.flatten(), dim=0).reshape(attn_map.shape)

    return attn_map

def get_grid_step_size(H, W, uniform = True):
    """
    Computes the spatial step size (grid spacing) for consistent divergence and curl calculations."
    """
    delta_x = (W - 1) / 2
    delta_y = delta_x if uniform else (H - 1) / 2

    return delta_x, delta_y

def generate_grid(H, W, normalize=False, aspect_aware=False):
    """
    Generates a grid with unit spacing and centered at (0,0).
    """
    x_coords = (torch.arange(W, dtype=torch.float32) - (W - 1) / 2).view(1, W).expand(H, W)
    y_coords = (torch.arange(H, dtype=torch.float32) - (H - 1) / 2).view(H, 1).expand(H, W)

    if normalize:
        delta_x, delta_y = get_grid_step_size(H, W, uniform = aspect_aware)
        x_coords = x_coords / delta_x
        y_coords = y_coords / delta_y

    return torch.stack([x_coords, y_coords], dim=-1)  # Shape (H, W, 2)

def distance_to_point(positions, point, method="manhattan"):
    """ 
    Compute distances to a point 
    """
    displacement = positions - point
    if method == "manhattan":
        distances = torch.sum(torch.abs(displacement), dim=-1)  # Scaled L1 norm
    elif method == "euclidean":
        distances = torch.sqrt(torch.sum(displacement**2, dim=-1))
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'manhattan' or 'euclidean'")

    return distances

def distance_to_line(positions, line_normal, line_point=None, signed=True):
    """ 
    Compute distance to a line.
    """
    line_point = torch.Tensor([0, 0]) if line_point is None else line_point
    line_normal = normalize_vector(line_normal)                    
    distances = torch.sum(line_normal * (positions - line_point), dim=-1)
    if not signed:
        distances = torch.abs(distances)

    return distances

def normalize_vector(vector, eps=1e-8):
    """ 
    Normalize vector
    """
    vector = vector.to(dtype=torch.float32) 
    norm = torch.norm(vector, p=2)
    if norm < eps:
        norm = norm + eps # Avoid division by zero
    vector = vector / norm

    return vector

def vector2normal(vector):
    """ 
    Generate perpendicular normal vectorl.
    """
    return normalize_vector(torch.tensor([-vector[1], vector[0]]))

def get_standard_normal(type="horizontal", H=None, W=None):
    """
    Generates a standard normal vector for common reference lines.

    Args:
        type (str): Type of normal vector. Options:
            - "horizontal"  -> (1, 0)  (Left to Right)
            - "vertical"    -> (0, 1)  (Top to Bottom)
            - "left_diag"   -> (-1, 1) (↙↗ Diagonal)
            - "right_diag"  -> (1, 1)  (↖↘ Diagonal)
    """
    H = H if H is not None else 1
    W = W if W is not None else 1

    if type == "horizontal":
        return torch.tensor([0, 1], dtype=torch.float32)
    elif type == "vertical":
        return torch.tensor([1, 0], dtype=torch.float32)
    elif type == "left_diag":
        normal = torch.tensor([H, W], dtype=torch.float32)
    elif type == "right_diag":
        normal = torch.tensor([-H, W], dtype=torch.float32)
    else:
        raise ValueError(f"Invalid type '{type}'. Choose from 'horizontal', 'vertical', 'left_diag', 'right_diag'.")
    
    return normalize_vector(normal)

def get_grid_step_size(H, W, uniform=True):
    """
    Computes the spatial step size (grid spacing) for consistent divergence and curl calculations."
    """
    delta_x = (W - 1) / 2
    delta_y = delta_x if uniform else (H - 1) / 2

    return delta_x, delta_y

def generate_grid(H, W, centered=False, grid_aspect="auto"):
    """
    Generates a 2D coordinate grid with optional centering and aspect-aware normalization.
    """
    y_coords, x_coords = torch.meshgrid(torch.arange(H, dtype=torch.float32) + 0.5,
                                        torch.arange(W, dtype=torch.float32) + 0.5,
                                        indexing="ij")
    if centered:
        x_coords = x_coords - W / 2
        y_coords = y_coords - H / 2

    if grid_aspect == "equal":
        x_coords = x_coords / max(H, W)
        y_coords = y_coords / max(H, W)
    elif grid_aspect == "scaled":
        x_coords = x_coords / W
        y_coords = y_coords / H

    return torch.stack([x_coords, y_coords], dim=-1)  # Shape (H, W, 2)


def get_filter_kernels(filter="central", dtype=torch.float32, device="cpu"):
    """
    Returns the correct kernel for the given filter type.
    """
    if filter == "central":
        kernel_dx = torch.tensor([[-0.5, 0, 0.5]], dtype=dtype, device=device)
        kernel_dy = torch.tensor([[-0.5], [0], [0.5]], dtype=dtype, device=device)
    elif filter == "sobel":
        kernel_dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
        kernel_dy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)
    elif filter == "prewitt":
        kernel_dx = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=dtype, device=device)
        kernel_dy = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=dtype, device=device)
    elif filter == "scharr":
        kernel_dx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=dtype, device=device)
        kernel_dy = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported filter operator: {filter}. Choose from 'central', 'sobel', 'prewitt', 'scharr'.")

    return kernel_dx.unsqueeze(0).unsqueeze(0), kernel_dy.unsqueeze(0).unsqueeze(0)  # Ensure shape (1, 1, kH, kW)

def kernel2padding(kernel):
    """
    Computes correct padding dimension for kernel.
    """
    H, W = kernel.shape[-2:]
    top = (H - 1) // 2
    bottom = top if H % 2 == 1 else top + 1
    left = (W - 1) // 2
    right = left if W % 2 == 1 else left + 1

    return (left, right, top, bottom)

def compute_gradients(attn_map, filter="sobel", dtype=torch.float32):
    """
    Computes gradients using Sobel filters.
    """
    attn_map = attn_map.to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
    kernel_dx, kernel_dy = get_filter_kernels(filter=filter, dtype=dtype)

    grad_x = F.conv2d(F.pad(attn_map, kernel2padding(kernel_dx), mode='reflect'), kernel_dx)
    grad_y = F.conv2d(F.pad(attn_map, kernel2padding(kernel_dy), mode='reflect'), kernel_dy)

    return torch.cat([grad_x, grad_y], dim=1) # Shape (1, 2, H, W)

def compute_divergence_and_curl(gradients, dtype=torch.float32, scale_to_grid=True):
    """
    Computes divergence and curl from gradients using grouped convolutions.
    """
    # Define convolution kernels
    kernel_dx, kernel_dy = get_filter_kernels(filter="central", dtype=dtype)

    # Stack kernels correctly: (2, 1, kH, kW) for group conv
    stacked_dx = torch.cat([kernel_dx, kernel_dx], dim=0)  # Shape: (2, 1, 1, 3)
    stacked_dy = torch.cat([kernel_dy, kernel_dy], dim=0)  # Shape: (2, 1, 3, 1)

    # Apply grouped convolution to compute all four partial derivatives
    partials_dx = F.conv2d(F.pad(gradients, kernel2padding(stacked_dx), mode='replicate'), stacked_dx, groups=2).squeeze(0)  # Shape: (1, 2, H, W)
    partials_dy = F.conv2d(F.pad(gradients, kernel2padding(stacked_dy), mode='replicate'), stacked_dy, groups=2).squeeze(0)  # Shape: (1, 2, H, W)

    if scale_to_grid:
        delta_x, delta_y = get_grid_step_size(gradients.shape[-2], gradients.shape[-1], uniform=True)
        partials_dx = partials_dx / delta_x
        partials_dy = partials_dy / delta_y

    # Compute divergence: div(F) = dF_x/dx + dF_y/dy
    divergence = partials_dx[0] + partials_dy[1]  # Shape: (H, W)

    # Compute curl: curl(F) = dF_y/dx - dF_x/dy
    curl = partials_dx[1] - partials_dy[0]  # Shape: (H, W)

    return divergence, curl

def compute_map_centroid(attn_map, positions, percentile=100, keep_aspect=True):
    """
    Compute centroid.
    """
    attn_map = attn_map.to(dtype=torch.float32)
    if percentile < 100:
        threshold = torch.quantile(attn_map, percentile / 100.0)  # Compute threshold value
        mask = attn_map >= threshold  # Create binary mask
        attn_map = attn_map * mask  # Zero out non-salient pixels
    moments = torch.sum(attn_map.unsqueeze(-1) * positions, dim=(0,1))
    centroid = moments / torch.sum(attn_map)

    return centroid

def compute_flow_around_point(gradients, positions, point):
    """
    Computes radial and angular flow relative to the centroid in normalized coordinates.
    """
    # Compute normalized displacement
    displacement = positions - point  # (H, W, 2)
    norm = torch.norm(displacement, dim=-1, keepdim=True) + 1e-8
    unit_displacement = displacement / norm  # Normalize to unit vectors

    # Compute radial and angular flow
    gradients = gradients.unsqueeze(0)

    radial_flow = (torch.stack((grad_x, grad_y), dim=-1) * unit_displacement).sum(dim=-1)
    angular_flow = (torch.stack((grad_x, grad_y), dim=-1) * torch.stack((-unit_displacement[..., 1], unit_displacement[..., 0]), dim=-1)).sum(dim=-1)

    return radial_flow, angular_flow

def balance_measures(attn_map, positions, sigma = 1, percentile=100):
    """
    MEasure balance.
    """
    H, W = attn_map.shape
    attn_map = attn_map.to(dtype=torch.float32)
    centroid = compute_map_centroid(attn_map, positions, percentile=percentile)
    d_VB = torch.sum(torch.abs(centroid))
    e_VB = gaussian_weighting(d_VB, sigma=sigma)

    vertical = get_standard_normal(type="vertical", H=H, W=W)
    angular_y = torch.sum(attn_map * distance_to_line(positions, vertical, signed=True))

    return centroid, e_VB, angular_y

def gaussian_weighting(distances, sigma=1):
    """ 
    Gaussian. 
    """
    return torch.exp(- (distances ** 2) / (2 * sigma))

def visualise_attn(attn_map, centroid=None, cmap='Blues'):
    """ 
    attention
    """
    plt.figure(figsize=(3, 4))
    plt.imshow(attn_map, cmap=cmap)
    plt.colorbar()
    plt.axis("off")
    # Plot centroid if provided
    if centroid is not None:
        plt.scatter(centroid[0], centroid[1], color='red', marker='+', s=100, linewidths=2, label="Centroid")

    plt.show()

def rot_lines(H, W):
    positions = generate_grid(H, W)
    horizontal_normal = get_standard_normal(type="horizontal")
    vertical_normal = get_standard_normal(type="vertical")

    H1_distances = distance_to_line(positions, horizontal_normal, H / 3)
    H2_distances = distance_to_line(positions, horizontal_normal, 2 * H / 3)
    H_distances = torch.min(H1_distances, H2_distances)

    V1_distances = distance_to_line(positions, vertical_normal, W / 3)
    V2_distances = distance_to_line(positions, vertical_normal, 2 * W / 3)
    V_distances = torch.min(V1_distances, V2_distances)

    distances = H_distances + V_distances
    distances = gaussian_weighting(distances)
    print("min =", distances.min().item(), "max =", distances.max().item(), "mean =", distances.mean().item())
    visualise_attn(distances)

    return distances

def rot_points(H, W):
    positions = generate_grid(H, W, normalize=True, keep_aspect=False)
    dist_1 = distance_to_point(positions, torch.tensor([-1/3, 1/3]))
    dist_2 = distance_to_point(positions, torch.tensor([1/3, 1/3]))
    dist_3 = distance_to_point(positions, torch.tensor([1/3, -1/3]))
    dist_4 = distance_to_point(positions, torch.tensor([-1/3, -1/3]))

    distances = torch.min(torch.stack([dist_1, dist_2, dist_3, dist_4], dim=0), dim=0).values

    return distances

def symmetry_mse(attn_map, sigma=1):
    mirror = torch.flip(attn_map, dims=[1])  # Flip horizontal
    score = torch.mean((attn_map - mirror)**2)
    return score

def symmetry_gaussian(attn_map):
    mirror = torch.flip(attn_map, dims=[1])  # Flip horizontal
    weight = gaussian_weighting((attn_map - mirror)**2, torch.var(attn_map**2)/10)
    print("min =", weight.min().item(), "max =", weight.max().item(), "mean =", weight.mean().item())
    visualise_attn(weight, cmap='hot')
    score = torch.mean(weight)
    return score


def plot_image(image_path, axes, third_lines=True):
    """
    Plots an image on the given axes with an option to overlay third lines.
    """
    img = mpimg.imread(image_path)  # Load image as a NumPy array
    axes.imshow(img)
    axes.axis("off")
    
    if third_lines:
        height, width = img.shape[:2]  # Extract dimensions
        # Draw vertical third lines
        axes.axvline(x=width / 3, color='red', linestyle='--')
        axes.axvline(x=2 * width / 3, color='red', linestyle='--')
        # Draw horizontal third lines
        axes.axhline(y=height / 3, color='red', linestyle='--')
        axes.axhline(y=2 * height / 3, color='red', linestyle='--')


# APPROACHES TO CONSIDER
# Divergence & curl → Vector field balance, spread, and rotational asymmetry.