"""
Implemntation of SSIM and SSIM3D loss in python
"""

import torch
import torch.nn.functional as F

# -------------------------------------------------------------------------------------------------
# Create windows

def gaussian(size, sigma):
    """
    Create the 1D gauss kernel given size and sigma
    """
    coords = torch.arange(size, dtype=torch.float) - size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma=1.5):
    """
    Creates a 2D gauss kernel and expands to the number of channels
    """
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def create_window_3D(window_size, channel, sigma=1.5):
    """
    Creates a 3D gauss kernel and expands to the number of channels
    """
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

# -------------------------------------------------------------------------------------------------
# Compute scores

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).mean(1)

# -------------------------------------------------------------------------------------------------
# Neat wrappers to handle devices and arbitrary sizes

class SSIM(torch.nn.Module):
    """
    SSIM for 2D images
    """
    def __init__(self, window_size=11, channel=1, size_average=True, device='cpu'):
        """
        @args:
            - window_size (int, odd number): size of square window filter
            - channel (int): expected channels of the images
            - size_average (bool): whether to average over all or over batch
            - device (torch.device): device to run on
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.window = self.window.to(device=device, dtype=torch.float32)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class SSIM3D(torch.nn.Module):
    """
    SSIM3D for 3D images
    """
    def __init__(self, window_size=11, channel=1, size_average=True, device='cpu'):
        """
        @args:
            - window_size (int, odd number): size of square window filter
            - channel (int): expected channels of the images
            - size_average (bool): whether to average over all or over batch
            - device (torch.device): device to run on
        """
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window_3D(window_size, self.channel)
        self.window = self.window.to(device=device, dtype=torch.float32)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

# -------------------------------------------------------------------------------------------------
# Testing

def tests():

    B,T,C,H,W = 4,8,3,32,32
    window_size = 7
    sigma=1.5

    window2D = create_window(window_size=window_size, channel=C, sigma=sigma)

    im2D_1 = torch.randn(B,C,H,W)
    im2D_2 = torch.randn(B,C,H,W)

    ssim_1 = _ssim(im2D_1, im2D_1, window=window2D, window_size=window_size, channel=C)
    assert ssim_1==1

    ssim_2 = _ssim(im2D_2, im2D_2, window=window2D, window_size=window_size, channel=C)
    assert ssim_2==1

    ssim_3 = _ssim(im2D_1, im2D_2, window=window2D, window_size=window_size, channel=C)
    assert 0<=ssim_3<=1

    ssim_4 = _ssim(im2D_2, im2D_1, window=window2D, window_size=window_size, channel=C)
    assert 0<=ssim_4<=1

    assert ssim_3==ssim_4

    print("Passed ssim2D")

    window3D = create_window_3D(window_size=window_size, channel=C, sigma=sigma)

    im3D_1 = torch.randn(B,C,T,H,W)
    im3D_2 = torch.randn(B,C,T,H,W)

    ssim_5 = _ssim_3D(im3D_1, im3D_1, window=window3D, window_size=window_size, channel=C)
    assert ssim_5==1

    ssim_6 = _ssim_3D(im3D_2, im3D_2, window=window3D, window_size=window_size, channel=C)
    assert ssim_6==1

    ssim_7 = _ssim_3D(im3D_1, im3D_2, window=window3D, window_size=window_size, channel=C)
    assert 0<=ssim_7<=1

    ssim_8 = _ssim_3D(im3D_2, im3D_1, window=window3D, window_size=window_size, channel=C)
    assert 0<=ssim_8<=1

    assert ssim_7==ssim_8

    print("Passed ssim3D")

    print("Passed all tests")

if __name__=="__main__":
    tests()
