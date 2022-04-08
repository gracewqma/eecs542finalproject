import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import math

# differentiable augmentation suite

# start with scratch implementation of each augmentation used in author's paper along with HSV implementation
class AugmentClass:
    
    def __init__(self, policy='', channels_first=True):
        self.policy = policy
        self.channels_first = channels_first

        self.saved_randoms = {}
        # for p in policy.split(','):
        #     self.augment_fns.append(self.AUGMENT_FNS[p])

    # def __call__(self, x):
    #     if self.channels_first:
    #         x = x.permute(0, 3, 1, 2)
    #     for f in self.augment_fns:
    #         x = f(x)
    #     if not self.channels_first:
    #         x = x.permute(0, 2, 3, 1)
    #     return x

    def augment(self, x, use_saved_randoms =False, channels_first=True):
        if self.policy:
            if not channels_first:
                x = x.permute(0, 3, 1, 2)
            for p in self.policy.split(','):
                for f in self.AUGMENT_FNS[p]:
                    x = f(self, x,use_saved_randoms)
            if not channels_first:
                x = x.permute(0, 2, 3, 1)
            x = x.contiguous()
        return x


    def brightness_orig(self, input, use_saved_randoms =False, shift_amount=0.5, distribution='uniform'):
        # get brightness value change per image in batch
        batch_size = input.shape[0]
        if use_saved_randoms:
            shift_vals = self.saved_randoms['brightness']
        else:
            shift_vals = torch.rand(batch_size, dtype=input.dtype, device=input.device) - shift_amount
            self.saved_randoms["brightness"] = shift_vals
        return input + torch.reshape(shift_vals, (batch_size, 1, 1, 1))

                
    def saturation(self, input,use_saved_randoms =False):
        input_mean = input.mean(dim=1, keepdim=True)
        if use_saved_randoms:
            shift_vals = self.saved_randoms['saturation']
        else:
            shift_vals = (torch.rand(input.size(0), 1, 1, 1, dtype=input.dtype, device=input.device) * 2)
            self.saved_randoms["saturation"] = shift_vals
        input = (input - input_mean) * shift_vals + input_mean
        return input
        

    def contrast(self,input, use_saved_randoms =False):
        # get mean over each dimension
        input_mean = input.mean(dim=[1, 2, 3], keepdim=True)
        if use_saved_randoms:
            shift_factor = self.saved_randoms['contrast']
        else:
            shift_factor = torch.rand(input.size(0), 1, 1, 1, dtype=input.dtype, device=input.device) + 0.5
            self.saved_randoms["contrast"] = shift_factor
        result = input * shift_factor + input_mean * (1-shift_factor)
        # result = torch.clamp(result, min=-1, max=1) 
        return result



    def hflip(self,input, use_saved_randoms =False, p=0.5):
        if use_saved_randoms:
            rand = self.saved_randoms['hflip']
        else:
            rand = torch.rand(1, dtype=input.dtype, device=input.device)
            self.saved_randoms["hflip"] = rand
        if  rand < p:
            w = input.shape[-1]
            return input[..., torch.arange(w - 1, -1, -1, device=input.device)]
        return input

    def rotation(self, input,use_saved_randoms =False):
        # sample value between [-1/6pi, 1/6pi]
        batch_size = input.shape[0]
        if use_saved_randoms:
            theta = self.saved_randoms['rotation']
        else:
            theta = ((1/3)*math.pi) * torch.rand(batch_size, 1, 1, dtype=input.dtype, device=input.device) - (1/6)*math.pi
            self.saved_randoms["rotation"] = theta
        first_row = torch.cat((torch.cos(theta), -torch.sin(theta), torch.zeros((batch_size, 1, 1), device=input.device)), dim=2)
        second_row = torch.cat((torch.sin(theta), torch.cos(theta), torch.zeros((batch_size, 1, 1), device=input.device)), dim=2)
        rot_mats = torch.cat((first_row, second_row), dim=1)

        grid = F.affine_grid(rot_mats, input.size())
        input = F.grid_sample(input, grid)

        return input

    # def blur(input):
    #     return transforms.GaussianBlur(5, sigma=(0.1, 2.0))(input)

    # def sharpen(input, sharpen_amount=2, probability=0.3):
    #     return transforms.RandomAdjustSharpness(sharpen_amount, probability)(input)




    AUGMENT_FNS = {
        'color': [brightness_orig, saturation, contrast],
        # 'translation': [rand_translation],
        # 'cutout': [rand_cutout],
        # 'gaussian': [blur, sharpen],
        'affine': [rotation, hflip],
    }