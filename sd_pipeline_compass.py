from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from attention import AttentionStore

class CompASSPipeline(StableDiffusionPipeline):
    """
    Initialize the CompASSPipeline, inheriting from StableDiffusionPipeline.
    """
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        requires_safety_checker=False,
    ):

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor
        )

        # Initialise resolution default variables
        self.default_output_resolution = None # Default output resolution for model
        self.unet_downsample_factor = None # UNet downsampling factor
        self.total_downsample_factor = None # Total VAE+UNet downscaling
        
        # Load resolution defaults
        self.get_resolution_defaults()

        # Additional components for attention tracking
        self.attn_store = AttentionStore()
        self.layer_metadata = defaultdict(list)
        self.hooks = []
        self.setup_hooks()
        self.diffused_images = []

        self.height = None
        self.width = None
        self.empty_embeds = None
        self.get_empty_embeddings()

    def get_resolution_defaults(self):
        """
        Compute and store default model parameters.
        """
        self.default_output_resolution = self.unet.config.sample_size * self.vae_scale_factor
        self.unet_downsample_factor = 2**(len(self.unet.config.block_out_channels)-1)
        self.total_downsample_factor = self.unet_downsample_factor * self.vae_scale_factor         

    def setup_hooks(self):
        """
        Set up hooks for cross-attention (attn2) and self-attention (attn1) layers.
        """
        self.clear_hooks()
        for layer_name, module in self.unet.named_modules():
            if hasattr(module, "is_cross_attention") and module.is_cross_attention:
                block_type, level, instance = sd_utils.parse_layer_name(layer_name)
                layer_key = (block_type, level, instance)
                self.layer_metadata[layer_key].append(layer_name)
                self.hooks.append(module.register_forward_hook(self._hook_fn(layer_key)))
        print(f"Number of hooks initialised: {len(self.hooks)}")

    def _hook_fn(self, layer_key):
        """
        Hook function to capture attention scores.
        """
        def hook(module, input, output):
            try:
                query = module.to_q(input[0])
                key = module.to_k(self.empty_embeddings)
                # key = module.to_k(self.text_embeddings.chunk(2, dim=0)[1] if is_cross else input[0])
                attn_probs = module.get_attention_scores(query, key)
                attn_maps = self.attn_store.reshape_attention_map(attn_probs, self.height, self.width)
                self.attn_store.store(attn_maps)
            except Exception as e:
                print(f"Error processing attention scores for layer {layer_key}: {e}")
        return hook

    def print_hook_metadata(self):
        print(f"Total Cross-Attention Layers: {len(self.layer_metadata)}")
        print("\nCross-Attention Layers:")
        for layer_key, layer_names in self.layer_metadata.items():
            print(f"{layer_key}: {layer_names}")  # Print the list directly

    def clear_hooks(self):
        """
        Remove all hooks from the model.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset(self):
        """
        Reset stored latents, images, and attention maps.
        """
        self.diffused_images.clear()
        self.attn_store.reset()

    def get_empty_embeddings(self, prompt="", batch_size=1):
        """
        Tokenize the prompt and get text embeddings. 
        """
        self.empty_embeds = self.encode_prompt(prompt, self.device, batch_size, False)
        # Output is tuple 
        print("Initiatilized empty embeddings")

    def image2latent(self, image, timesteps, seed=42):
        """
        Prepare latents from an image or random noise.
        """
        if image.shape[0] != len(timesteps):
            if image.shape[1] == 1:
                image = image.repeat(len(timesteps), 1, 1, 1)
            else:
                raise ValueError(f"Image shape {image.shape} does not match timesteps {timesteps}")

        latents = self.vae.encode(image).latent_dist.mean * self.vae.config.scaling_factor
        # Set seed for reproducibility 
        generator = torch.Generator(device=self.device).manual_seed(seed)
        batch_size, num_channels, height, width = latents.shape
        noise = sd_utils.init_latent(batch_size, num_channels, height, width, generator=generator, dtype=pipe.dtype)  # Generate noise deterministically
        # Apply timestep-dependent noise across batch channel
        latents = self.scheduler.add_noise(latents, noise, timesteps)

    def extract_reference_attn_maps(self, image, timesteps, seed=42):
        batch_size = len(timesteps)
        if image.shape[0] != batch_size:
            if image.shape[0] == 1:
                image = image.repeat(batch_size, 1, 1, 1)
            else:
                raise ValueError(f"Image shape {image.shape} does not match timesteps {batch_size}")
        print(self.empty_embeds[0].shape)

        if self.empty_embeds[0].shape[0] != batch_size:
            self.empty_embeds = (self.empty_embeds[0].repeat(batch_size, 1, 1), self.empty_embeds[1])

        latents = self.image2latent(image, timesteps, seed)
        with torch.no_grad():
            unet_output = self.pipe.unet(latents, timesteps, self.text_embeddings, return_dict=True)
            noise_pred = unet_output["sample"]
            latents = self.pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]