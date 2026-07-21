# Model design

The model learns a conditional distribution of 1,024-point chair surfaces from
one LiDAR scan and its sensor pose.

| Choice | Basis |
|---|---|
| Straight conditional flow path | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) |
| Pointwise nearest-neighbor target alignment and Chamfer supervision | [LiFlow](https://arxiv.org/abs/2602.02232) |
| Point-transformer conditioning for completion | [MedPCFM](https://arxiv.org/abs/2606.24433) |
| Source point supplied to the velocity field | [mmWaveFlow](https://openaccess.thecvf.com/content/CVPR2026/html/Su_mmWaveFlow_Unified_Enhancement_and_Generation_of_mmWave_Human_Point_Clouds_CVPR_2026_paper.html) |
| Condition tokens gather from moving points, mix globally, and project back | [NOVA3R](https://openreview.net/forum?id=c0QRZMKwSb) |
| Adaptive LayerNorm with zero-initialized residual gates | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) |
| Local processing before global point attention | [LitePT](https://openaccess.thecvf.com/content/CVPR2026/html/Yue_LitePT_Lighter_Yet_Stronger_Point_Transformer_CVPR_2026_paper.html) |

Observed points carry object-frame position, ray direction, range, and
sensor-relative displacement. The sensor token carries position, forward
direction, hit fraction, and range statistics. The condition encoder produces
257 tokens. Eight latent-mediated flow blocks predict one velocity per moving
point. Training uses flow regression, symmetric Chamfer distance, and one-sided
agreement with measured points. Sampling uses eight Heun steps.

The model has 26.8 million parameters at width 256. AdamW uses a cosine schedule,
bfloat16 arithmetic, gradient clipping, and an exponential moving average. The
default run contains 100,000 updates with batch size 32 on one A100.
