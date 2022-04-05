import torch
import torch.nn.functional as F
import math

# differentiable augmentation suite

# start with scratch implementation of each augmentation used in author's paper along with HSV implementation

def Augment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def brightness_orig(input, shift_amount=0.5, distribution='uniform'):
    # get brightness value change per image in batch
    batch_size = input.shape[0]
    shift_vals = torch.rand(batch_size, dtype=input.dtype, device=input.device) - shift_amount
    return input + torch.reshape(shift_vals, (batch_size, 1, 1, 1))

def hue(input):
    # convert hsv
    batch_size = input.shape[0]
    hsv = rgb2hsv(input)
    h,s,v = torch.chunk(hsv, chunks =3, dim = -3)
    # sample value between [-.25pi, .25pi]
    shift_amount = (0.5*math.pi) * torch.rand(batch_size, 1, 1, 1, dtype=input.dtype, device=input.device) - 0.25*math.pi
    # mod by 2pi to get value
    # differentiable?
    h_out = torch.fmod(h + shift_amount, 2 * math.pi)
    hsv_out = torch.cat([h_out, s, v], dim=-3)
    # convert back to rgb
    rgb_out = hsv2rgb(hsv_out)
    return rgb_out

              
def saturation(input):
    # convert to hsv
    # batch_size = input.shape[0]
    # # sample value between [0, 2]
    # shift_amount = torch.rand(batch_size, 1, 1, 1, dtype=input.dtype, device=input.device) * 2
    # hsv = get_hsv(input)
    # h,s,v = torch.chunk(hsv, chunks =3, dim = -3)
    # # adjust saturation from grayscale to double saturation
    # s_out = s * shift_amount
    # hsv_out = torch.cat([h, s_out, v], dim=-3)
    # # convert back to rgb
    # rgb_out = get_rgb_from_hsv(hsv_out)
    # return rgb_out

    input_mean = input.mean(dim=1, keepdim=True)
    input = (input - input_mean) * (torch.rand(input.size(0), 1, 1, 1, dtype=input.dtype, device=input.device) * 2) + input_mean
    return input
    

def contrast(input):
    # get mean over each dimension
    input_mean = input.mean(dim=[1, 2, 3], keepdim=True)
    shift_factor = torch.rand(input.size(0), 1, 1, 1, dtype=input.dtype, device=input.device) + 0.5
    result = input * shift_factor + input_mean * (1-shift_factor)
    # result = torch.clamp(result, min=-1, max=1) 
    return result


# from diff aug paper
def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x

# from diff aug paper
def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def hflip(input, p=0.5):
    if torch.rand(1, dtype=input.dtype, device=input.device)  < p:
        w = input.shape[-1]
        return input[..., torch.arange(w - 1, -1, -1, device=input.device)]
    return input

def rotation(input):
    # sample value between [-1/6pi, 1/6pi]
    batch_size = input.shape[0]
    theta = ((1/3)*math.pi) * torch.rand(batch_size, 1, 1, dtype=input.dtype, device=input.device) - (1/6)*math.pi
    first_row = torch.cat((torch.cos(theta), -torch.sin(theta), torch.zeros((batch_size, 1, 1), device=input.device)), dim=2)
    second_row = torch.cat((torch.sin(theta), torch.cos(theta), torch.zeros((batch_size, 1, 1), device=input.device)), dim=2)
    rot_mats = torch.cat((first_row, second_row), dim=1)

    grid = F.affine_grid(rot_mats, input.size())
    input = F.grid_sample(input, grid)

    return input




# # rgb2hsv taken from Kornia library
# def rgb2hsv(input):
#     max_rgb, argmax_rgb = input.max(-3)
#     min_rgb, argmin_rgb = input.min(-3)
#     deltac = max_rgb - min_rgb

#     v = max_rgb
#     s = deltac / (max_rgb + 1e-8)

#     deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
#     rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - input), dim=-3)

#     h1 = (bc - gc)
#     h2 = (rc - bc) + 2.0 * deltac
#     h3 = (gc - rc) + 4.0 * deltac

#     h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
#     h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
#     h = (h / 6.0) % 1.0
#     h = 2. * math.pi * h  # we return 0/2pi output

#     return torch.stack((h, s, v), dim=-3)

# def hsv2rgb(input):
#     h: torch.Tensor = input[..., 0, :, :] / (2 * math.pi)
#     s: torch.Tensor = input[..., 1, :, :]
#     v: torch.Tensor = input[..., 2, :, :]

#     hi: torch.Tensor = torch.floor(h * 6) % 6
#     f: torch.Tensor = ((h * 6) % 6) - hi
#     one: torch.Tensor = torch.tensor(1.0, device=input.device, dtype=input.dtype)
#     p: torch.Tensor = v * (one - s)
#     q: torch.Tensor = v * (one - f * s)
#     t: torch.Tensor = v * (one - (one - f) * s)

#     hi = hi.long()
#     indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
#     out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
#     out = torch.gather(out, -3, indices)

#     return out


AUGMENT_FNS = {
    'color': [brightness_orig, saturation, contrast],
    'translation': [rand_translation, rotation, hflip],
    'cutout': [rand_cutout],
}