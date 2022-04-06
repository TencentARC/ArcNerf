# Base_modules
Modules like embedder, implicit function, radiance field, etc.
## embedder
Embed inputs like xyz/view_dir into higher dimension using periodic funcs(sin, cos).
- output_dim = (inputs_dim * len(periodic_fns) * N_freq + include_input * inputs_dim)
- log_sampling: if True, use log factor sin(2**N * x). Else use scale factor sin(N * x).
## activation
- Sine: sin() for activation.
## linear
- DenseLayer: Linear with custom activation function.
- SirenLayer: Linear with sin() activation and respective initialization.
## implicit network
implicit network for mapping xyz -> sigma/sdf/rgb, etc.
### GeoNet
Multiple DenseLayer/SirenLayer.
### RadianceNet
Multiple DenseLayer/SirenLayer.


## Nerf
Nerf model with single forward, and hierarchical sampling.
It is combination of GeoNet and RadianceNet.
