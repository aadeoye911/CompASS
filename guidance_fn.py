# if target_heatmap is None:
#         raise ValueError("You must provide a target_heatmap of shape [1, H, W] or [B, H, W].")

#     if target_heatmap.shape[0] == 1:
#         target_heatmap = target_heatmap.expand(B, -1, -1)

#     target_heatmap = target_heatmap.to(device)
#     target_heatmap = target_heatmap / target_heatmap.sum(dim=(1, 2), keepdim=True)

#     # Compute KL divergence: D(P || Q)
#     eps = 1e-6
#     P = kde_map + eps
#     Q = target_heatmap + eps
#     loss = F.kl_div(P.log(), Q, reduction="batchmean")

#     return loss