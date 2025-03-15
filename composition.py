import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def normalize_map(map):
    min = map.min()
    max = map.max()
    if max == min:
        return torch.zeros_like(map) # Avoid division by zero
    map = (map - min) / (max - min)

    return map


def softmax_normalization(attn_map, temperature=1.0):
    attn_map = attn_map / temperature  # Optional: Adjust distribution sharpness
    attn_map = torch.nn.functional.softmax(attn_map.flatten(), dim=0).reshape(attn_map.shape)

    return attn_map


def generate_normalized_grid(H, W, keep_aspect=True):
    aspect = H / W if keep_aspect else 1
    x_coords, y_coords = torch.meshgrid(torch.linspace(-1, 1, steps=W),
                                        torch.linspace(-aspect, aspect, steps=H),
                                        indexing="xy")
     
    return torch.stack((x_coords, y_coords), dim=-1)    


def distance_to_point(positions, point, method="manhattan"):
    displacement = positions - point
    if method == "manhattan":
        distances = torch.sum(torch.abs(displacement), dim=-1)  # Scaled L1 norm
    elif method == "euclidean":
        distances = torch.sqrt(torch.sum(displacement**2, dim=-1))
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'manhattan' or 'euclidean'")

    return distances


def distance_to_line(positions, line_normal, line_point=None, signed=True):
    line_point = torch.Tensor([0, 0]) if line_point is None else line_point
    line_normal = normalize_vector(line_normal)                    
    distances = torch.sum(line_normal * (positions - line_point), dim=-1)
    if not signed:
        distances = torch.abs(distances)

    return distances


def normalize_vector(vector, eps=1e-8):
    vector = vector.to(dtype=torch.float32) 
    norm = torch.norm(vector, p=2)
    if norm < eps:
        norm = norm + eps # Avoid division by zero
    vector = vector / norm

    return vector


def vector2normal(vector):
    normal = torch.tensor([-vector[1], vector[0]])

    return normalize_vector(normal)


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

def compute_gradients(attn_map, keep_aspect=True):
    """
    Computes gradients using Sobel filters.
    """
    H, W = attn_map.shape
    attn_map = attn_map.to(dtype=torch.float32)  # Ensure float32

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=attn_map.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=attn_map.device).unsqueeze(0).unsqueeze(0)

    attn_map = attn_map.unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
    grad_x = F.conv2d(F.pad(attn_map, (1, 1, 1, 1), mode='replicate'), sobel_x)
    grad_y = F.conv2d(F.pad(attn_map, (1, 1, 1, 1), mode='replicate'), sobel_y)

    if keep_aspect:
        grad_x = grad_x * (2 / W)       # Scale by grid range [-1, 1] in x-direction
        grad_y = grad_y * (2*(H/W) / H) # Scale by grid range [-H/W, H/W] in y-direction

    gradients = torch.cat([grad_x, grad_y], dim=1) # Shape (1, 2, H, W)
    
    return gradients

def compute_divergence_and_curl(gradients):
    """
    Computes divergence and curl from gradients using grouped convolutions.
    """
    # Define convolution kernels
    kernel_dx = torch.tensor([[-0.5, 0, 0.5]], dtype=torch.float32, device=gradients.device).unsqueeze(0)  # (1, 3)
    kernel_dy = torch.tensor([[-0.5], [0], [0.5]], dtype=torch.float32, device=gradients.device).unsqueeze(0)  # (3, 1)

    # Stack kernels correctly: (2, 1, kH, kW) for group conv
    stacked_dx = torch.stack([kernel_dx, kernel_dx], dim=0)  # Shape: (2, 1, 1, 3)
    stacked_dy = torch.stack([kernel_dy, kernel_dy], dim=0)  # Shape: (2, 1, 3, 1)

    # Apply grouped convolution to compute all four partial derivatives
    partials_dx = F.conv2d(F.pad(gradients, (1, 1, 0, 0), mode='replicate'), stacked_dx, groups=2).squeeze(0)  # Shape: (1, 2, H, W)
    partials_dy = F.conv2d(F.pad(gradients, (0, 0, 1, 1), mode='replicate'), stacked_dy, groups=2).squeeze(0)  # Shape: (1, 2, H, W)

    # Compute divergence: div(F) = dF_x/dx + dF_y/dy
    divergence = partials_dx[0] + partials_dy[1]  # Shape: (H, W)

    # Compute curl: curl(F) = dF_y/dx - dF_x/dy
    curl = partials_dx[1] - partials_dy[0]  # Shape: (H, W)

    return divergence, curl

def get_focal_centroid(attn_map, keep_aspect=True):
    H, W = attn_map.shape
    attn_map = attn_map.to(dtype=torch.float32)
    positions = generate_normalized_grid(H, W, keep_aspect=keep_aspect)
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


def balance_measures(attn_map, keep_aspect=True, sigma=1):
    H, W = attn_map.shape
    attn_map = attn_map.to(dtype=torch.float32)
    # attn_map = normalize_map(attn_map)
    focal_strength = torch.max(attn_map) / torch.mean(attn_map)

    attn_map = softmax_normalization(attn_map, temperature=focal_strength)
    positions = generate_normalized_grid(H, W, keep_aspect=keep_aspect)
    attn_mass = torch.sum(attn_map)

    
    # Compute center of mass
    moments = torch.sum(attn_map.unsqueeze(-1) * focal_strength * positions, dim=(0,1))
    centroid = moments / attn_mass
    d_VB = torch.sum(torch.abs(centroid ))
    e_VB = gaussian_weighting(d_VB, sigma=sigma)

    # Compute moment of inertia
    # squared_distances = torch.sum(attn_map.unsqueeze(-1) * ((positions - centroid)** 2), dim=(0, 1))
    # variance = squared_distances / attn_mass
    # moment_of_inertia = torch.sum(squared_distances / (H * W))

    # Compute moment of inertia
    # right_diag = get_standard_normal(type="right_diag", H=H, W=W)
    # left_diag = get_standard_normal(type="left_diag", H=H, W=W)
    # angular_right = torch.sum(attn_map * distance_to_line(positions, right_diag, line_point=centroid, signed=True))
    # angular_left = torch.sum(attn_map * distance_to_line(positions, left_diag, line_point=centroid, signed=True))

    vertical = get_standard_normal(type="vertical", H=H, W=W)
    vertical_momentum = torch.sum(attn_map * distance_to_line(positions, vertical, line_point=centroid, signed=True))
    
    return centroid, e_VB, vertical_momentum

def gaussian_weighting(distances, sigma=1):
    """ 
    Gaussian. 
    """
    return torch.exp(- (distances ** 2) / (2 * sigma))


def visualise_attn(attn_map, centroid=None, cmap='Blues'):
    plt.figure(figsize=(3, 4))
    plt.imshow(attn_map, cmap=cmap)
    plt.colorbar()
    plt.axis("off")
    # Plot centroid if provided
    if centroid is not None:
        plt.scatter(centroid[0], centroid[1], color='red', marker='+', s=100, linewidths=2, label="Centroid")

    plt.show()


def rot_lines(H, W):
    positions = generate_normalized_grid(H, W)
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
    positions = generate_normalized_grid(H, W, keep_aspect=False)
    dist_1 = distance_to_point(positions, torch.tensor([-1/3, 1/3]))
    dist_2 = distance_to_point(positions, torch.tensor([1/3, 1/3]))
    dist_3 = distance_to_point(positions, torch.tensor([1/3, -1/3]))
    dist_4 = distance_to_point(positions, torch.tensor([-1/3, -1/3]))

    distances = torch.min(torch.stack([dist_1, dist_2, dist_3, dist_4], dim=0), dim=0).values
    distances = gaussian_weighting(distances)
    print("min =", distances.min().item(), "max =", distances.max().item(), "mean =", distances.mean().item())
    visualise_attn(distances)

    return distances


def symmetry_mse(attn_map, sigma=1):
    H, W = attn_map.shape
    attn_map = normalize_map(attn_map)
    mirror = torch.flip(attn_map, dims=[1])  # Flip horizontal
    score = torch.mean((attn_map - mirror)**2)
    return score

def symmetry_gaussian(attn_map):
    attn_map = normalize_map(attn_map)
    mirror = torch.flip(attn_map, dims=[1])  # Flip horizontal
    weight = gaussian_weighting((attn_map - mirror)**2, torch.var(attn_map**2)/10)
    print("min =", weight.min().item(), "max =", weight.max().item(), "mean =", weight.mean().item())
    visualise_attn(weight, cmap='hot')
    score = torch.mean(weight)
    return score


# APPROACHES TO CONSIDER
# Moments (centroid, variance, skewness) → Physics/Statistics balance.
# Fourier transform → Spatial balance, symmetry, rule-of-thirds.
# Divergence & curl → Vector field balance, spread, and rotational asymmetry.