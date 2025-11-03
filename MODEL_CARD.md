# Model Card — LightGAN-LD

## Intended Use
Research on LDCT reconstruction and benchmarking. **Not for clinical diagnosis**.

## Metrics (reference)
- LoDoPaB-CT: ~36.79 dB PSNR / 0.894 SSIM.
- NIH-AAPM-Mayo: ~35.87 dB PSNR / 0.898 SSIM.

## Efficiency
~5.2M params; ~28 ms / 256×256 slice on RTX 3090 (reference).

## Limitations
Distribution shift; extreme sparse views; GAN artifacts.

## Ethics
No PHI; dataset licenses apply.
