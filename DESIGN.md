# Model design

The model generates a query LiDAR observation from a current observation and
the relative query-sensor pose.

Each scan is represented internally on its native `32 x 32` ray grid. A cell
stores hit state and normalized range. This preserves misses, occlusion, and ray
identity while allowing generated states to be converted directly to 3D points.

The current scan and query state are embedded as `2 x 2` patches. Four
transformer blocks encode the current scan. Eight adaptive-LayerNorm transformer
blocks process the moving query state with self-attention and cross-attention to
the current scan. Query-ray origins and directions are expressed in the current
sensor frame, so the relative pose enters both geometrically and through the
block conditioning. The initial noise is not provided to the transformer.

Training uses a straight conditional flow path from Gaussian noise to the query
scan. Occupancy flow and return-range flow receive equal loss weight. Sampling
uses eight Heun steps. Checkpoint selection and early stopping use complete
rollouts, combining hit-mask intersection over union with return-range error.
AdamW, bfloat16, gradient clipping, cosine learning-rate decay, and exponential
moving averages follow the training setup.

Sources:

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
