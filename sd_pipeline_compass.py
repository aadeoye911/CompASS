
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

        # Additional components for attention tracking
        self.hooks = []
        self.setup_hooks()

        self.attention_store = AttentionStore()
        self.layer_metadata = defaultdict(list)
        self.decoded_images = []
        self.latents = None

        self.height = None
        self.width = None
        self.prompt_embeds = None

        self.setup_hooks()

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
                key = module.to_k(self.text_embeddings)
                # key = module.to_k(self.text_embeddings.chunk(2, dim=0)[1] if is_cross else input[0])
                attention_scores = module.get_attention_scores(query, key)
                self.attention_store.store(attention_scores)
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
        self.decoded_images.clear()
        self.attn_store.reset()

    def get_text_embeddings(self, prompt="", batch_size=1):
        """
        Tokenize the prompt and get text embeddings.
        """
        self.prompt_embeds = self.encode_prompt(prompt,
                                device=self.device,
                                num_images_per_prompt=batch_size,
                                do_classifier_free_guidance=False)

    def image2tensor(image):
        transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
        image_tensor = transform(image).unsqueeze(0).to(pipe.device, dtype=pipe.dtype)
        return image_tensor