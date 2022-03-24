import torch
import torch.nn.functional as F
import math

# differentiable augmentation suite

# start with scratch implementation of each augmentation used in author's paper along with HSV implementation

def brightness_orig(input, shift_amount, distribution='uniform'):
    # get brightness value change per image in batch
    batch_size = input.shape[0]
    shift_vals = torch.rand(batch_size, dtype=input.dtype, device=input.device) - shift_amount
    return input + torch.reshape(shift_vals, (batch_size, 1, 1, 1))

def hue(input):
    # convert hsv
    batch_size = input.shape[0]
    hsv = rgb2hsv(input)
    print(input.shape)
    print(hsv.shape)
    h,s,v = torch.chunk(hsv, chunks =3, dim = -3)
    print(h.shape)
    # sample value between [-pi, pi]
    shift_amount = (2*math.pi) * torch.rand(batch_size, 1, 1, 1, dtype=input.dtype, device=input.device) - math.pi
    print(shift_amount.shape)
    # mod by 2pi to get value
    # differentiable?
    h_out = torch.fmod(h + shift_amount, 2 * math.pi)
    hsv_out = torch.cat([h_out, s, v], dim=-3)
    # convert back to rgb
    rgb_out = hsv2rgb(hsv_out)
    return rgb_out

              
def saturation(input):
    # convert to hsv
    batch_size = input.shape[0]
    # sample value between [0, 2]
    shift_amount = torch.rand(batch_size, 1, 1, 1, dtype=input.dtype, device=input.device) * 2
    hsv = rgb2hsv(input)
    h,s,v = torch.chunk(hsv, chunks =3, dim = -3)
    # adjust saturation from grayscale to double saturation
    s_out = s * shift_amount
    hsv_out = torch.cat([h, s_out, v], dim=-3)
    # convert back to rgb
    rgb_out = hsv2rgb(hsv_out)
    return rgb_out
    
    
def contrast(input):
    # get mean over each dimension
    input_mean = input.mean(dim=[1, 2, 3], keepdim=True)
    shift_factor = torch.rand(input.size(0), 1, 1, 1, dtype=input.dtype, device=input.device) + 0.5
    result = input * shift_factor + input_mean * (1-shift_factor)
    return result


# rgb2hsv taken from Kornia library
def rgb2hsv(input):
    max_rgb, argmax_rgb = input.max(-3)
    min_rgb, argmin_rgb = input.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + 1e-8)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - input), dim=-3)

    h1 = (bc - gc)
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2. * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)

def hsv2rgb(input):
    h: torch.Tensor = input[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = input[..., 1, :, :]
    v: torch.Tensor = input[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=input.device, dtype=input.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out