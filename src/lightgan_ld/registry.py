def build_dataset(cfg, split="train"):
    name = cfg["data"]["name"]
    if name == "dummy":
        from .data.dummy import DummyPairs
        ds = DummyPairs(n=128 if split=="train" else 32, img_size=cfg["data"]["img_size"], sino_shape=tuple(cfg["data"]["sinogram_shape"]))
        return ds
    elif name == "lodopab":
        from .data.lodopab import LoDoPaBPair
        path = f'{cfg["data"]["root"]}/lodopab_{split}.h5'
        return LoDoPaBPair(path, resize=cfg["data"]["img_size"], augment=(split=="train"))
    elif name == "mayo":
        from .data.mayo import MayoPair
        path = f'{cfg["data"]["root"]}/mayo_{split}.h5'
        return MayoPair(path, resize=cfg["data"]["img_size"], augment=(split=="train"))
    else:
        raise ValueError(f"Unknown dataset: {name}")

def build_models(cfg, device="cuda"):
    from .models.generator import LightGANLDGenerator
    from .models.discriminator import PatchDiscriminator
    from .models.sinogram_encoder import SinogramEncoder
    G = LightGANLDGenerator(
        in_ch=cfg["model"]["generator"]["in_channels"],
        base=cfg["model"]["generator"]["base_channels"],
        num_down=cfg["model"]["generator"]["num_down"],
        use_ghost=cfg["model"]["generator"]["use_ghost"],
        use_condconv=cfg["model"]["generator"]["use_condconv"],
        use_eca=cfg["model"]["generator"]["use_eca"],
        experts=cfg["model"]["generator"]["condconv_experts"],
    ).to(device)
    D = PatchDiscriminator(
        in_ch=cfg["model"]["discriminator"]["in_channels"],
        base=cfg["model"]["discriminator"]["base_channels"],
        spectral_norm=cfg["model"]["discriminator"]["spectral_norm"]
    ).to(device)
    E = SinogramEncoder(
        in_ch=1, out_ch=1, channels=cfg["model"]["sinogram_encoder"]["channels"]
    ).to(device)
    return G, D, E
