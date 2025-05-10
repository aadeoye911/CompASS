import torch
import math
import torch.nn.functional as F

def minmax_normalization(attn_map):
    min = attn_map.amin(dim=(1, 2), keepdim=True)  # [B, 1, 1, 1]
    max = attn_map.amax(dim=(1, 2), keepdim=True)  # [B, 1, 1, 1]

    return (attn_map - min) / (max - min)

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
    """ 
    Compute distance to a line.
    """
    line_point = torch.Tensor([0, 0]) if line_point is None else line_point
    line_normal = normalize_vector(line_normal)                    
    distances = torch.sum(line_normal * (positions - line_point), dim=-1)
    if not signed:
        return torch.abs(distances)

    return distances

def normalize_vector(vector, eps=1e-8):
    vector = vector.to(dtype=torch.float32) 
    norm = torch.norm(vector, p=2)
    if norm < eps:
        norm = norm + eps # Avoid division by zero
    return vector / norm

def vector2normal(vector):
    return normalize_vector(torch.tensor([-vector[1], vector[0]]))

def get_standard_normal(type="horizontal", H=None, W=None):
    """
    Generates a standard normal vector for common reference lines.
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

def compute_centroids(attn_map, grid):
    grid = grid.unsqueeze(0)  # [1, H, W, 2]
    attn_map = minmax_normalization(attn_map)
    weighted_coords = attn_map * grid  # [B, H, W, 2]
    centroids = weighted_coords.sum(dim=(1, 2)) / attn_map.sum(dim=(1, 2))

    return centroids # [B, 2]

def centroids_to_kde(centroids, grid, sigma=1):
    batch_size, num_samples, _ = centroids.shape
    # Compute squared distance between each centroid and grid location
    diffs = (centroids.unsqueeze(2).unsqueeze(3) - grid)  # [B, N, H, W, 2]
    dists = (diffs ** 2).sum(dim=-1)  # [B, N, H, W]
    # Apply Gaussian kernel function
    weights = gaussian_weighting(dists, sigma)
    kde = weights.sum(dim=1) / (num_samples * (2 * math.pi * sigma**2))
    # Normalise to a valid PDF
    kde = kde / kde.sum(dim=(1, 2), keepdim=True) # [B, H, W]
    return kde 

def compute_torque(attn_map):
    H, W = attn_map.shape
    positions = generate_grid(H, W, grid_aspect="equal", centered=True)

    vertical = get_standard_normal("vertical")
    left_third = torch.Tensor([-1/6, 0])
    right_third = torch.Tensor([1/6, 0])

    left_torque = torch.sum(attn_map * distance_to_line(positions, vertical, left_third))
    right_torque = torch.sum(attn_map * distance_to_line(positions, vertical, right_third))
    
    return right_torque - left_torque

def gaussian_weighting(distances, sigma=1):
    return torch.exp(- (distances ** 2) / (2 * sigma**2))

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

# def z_normalization(attn_map):
#     """ 
#     Z normalization
#     """
#     mean = attn_map.mean()
#     std = attn_map.std()
#     # if torch.isclose(std, torch.tensor(0.0), atol=1e-8)
#     #     return torch.zeros_like(attn_map) # Avoid division by zero
#     return (attn_map - mean) / std

# def get_filter_kernels(filter="central", dtype=torch.float32, device="cpu"):
#     """
#     Returns the correct kernel for the given filter type.
#     """
#     if filter == "central":
#         kernel_dx = torch.tensor([[-0.5, 0, 0.5]], dtype=dtype, device=device)
#         kernel_dy = torch.tensor([[-0.5], [0], [0.5]], dtype=dtype, device=device)
#     elif filter == "sobel":
#         kernel_dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device)
#         kernel_dy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device)
#     elif filter == "prewitt":
#         kernel_dx = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=dtype, device=device)
#         kernel_dy = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=dtype, device=device)
#     elif filter == "scharr":
#         kernel_dx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=dtype, device=device)
#         kernel_dy = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=dtype, device=device)
#     else:
#         raise ValueError(f"Unsupported filter operator: {filter}. Choose from 'central', 'sobel', 'prewitt', 'scharr'.")

#     return kernel_dx.unsqueeze(0).unsqueeze(0), kernel_dy.unsqueeze(0).unsqueeze(0)  # Ensure shape (1, 1, kH, kW)

# def kernel2padding(kernel):
#     """
#     Computes correct padding dimension for kernel.
#     """
#     H, W = kernel.shape[-2:]
#     top = (H - 1) // 2
#     bottom = top if H % 2 == 1 else top + 1
#     left = (W - 1) // 2
#     right = left if W % 2 == 1 else left + 1

#     return (left, right, top, bottom)

# def compute_gradients(attn_map, filter="sobel", dtype=torch.float32):
#     """
#     Computes gradients using Sobel filters.
#     """
#     attn_map = attn_map.to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
#     kernel_dx, kernel_dy = get_filter_kernels(filter=filter, dtype=dtype)

#     grad_x = F.conv2d(F.pad(attn_map, kernel2padding(kernel_dx), mode='reflect'), kernel_dx)
#     grad_y = F.conv2d(F.pad(attn_map, kernel2padding(kernel_dy), mode='reflect'), kernel_dy)

#     return torch.cat([grad_x, grad_y], dim=1) # Shape (1, 2, H, W)

# def compute_divergence_and_curl(gradients, dtype=torch.float32, scale_to_grid=True):
    # """
    # Computes divergence and curl from gradients using grouped convolutions.
    # """
    # # Define convolution kernels
    # kernel_dx, kernel_dy = get_filter_kernels(filter="central", dtype=dtype)

    # # Stack kernels correctly: (2, 1, kH, kW) for group conv
    # stacked_dx = torch.cat([kernel_dx, kernel_dx], dim=0)  # Shape: (2, 1, 1, 3)
    # stacked_dy = torch.cat([kernel_dy, kernel_dy], dim=0)  # Shape: (2, 1, 3, 1)

    # # Apply grouped convolution to compute all four partial derivatives
    # partials_dx = F.conv2d(F.pad(gradients, kernel2padding(stacked_dx), mode='replicate'), stacked_dx, groups=2).squeeze(0)  # Shape: (1, 2, H, W)
    # partials_dy = F.conv2d(F.pad(gradients, kernel2padding(stacked_dy), mode='replicate'), stacked_dy, groups=2).squeeze(0)  # Shape: (1, 2, H, W)

    # if scale_to_grid:
    #     delta_x, delta_y = get_grid_step_size(gradients.shape[-2], gradients.shape[-1], uniform=True)
    #     partials_dx = partials_dx / delta_x
    #     partials_dy = partials_dy / delta_y

    # # Compute divergence: div(F) = dF_x/dx + dF_y/dy
    # divergence = partials_dx[0] + partials_dy[1]  # Shape: (H, W)

    # # Compute curl: curl(F) = dF_y/dx - dF_x/dy
    # curl = partials_dx[1] - partials_dy[0]  # Shape: (H, W)

    # return divergence, curl



# APPROACHES TO CONSIDER
# Divergence & curl → Vector field balance, spread, and rotational asymmetry.