from torchvision.models.vgg import vgg19
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from numpy.core import integer, empty, arange, asarray, roll
from numpy.core.overrides import array_function_dispatch, set_module
import numpy.fft as fft
import numpy as np


class CosineSimilarity(nn.Module):
    r"""Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along dim.
    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.
    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same shape as the Input1
        - Output: :math:`(\ast_1, \ast_2)`
    Examples::
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        cos = torch.nn.CosineSimilarity(dim=1)
        loss_cos = torch.mean(1 - cos(x1, x2))

        return loss_cos  # F.cosine_similarity(x1, x2, self.dim, self.eps)


class perceptual_loss(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(perceptual_loss, self).__init__()

        self.maeloss = torch.nn.L1Loss()
        vgg = vgg19(pretrained=True).cuda()

        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, X, Y):
        # print(X.shape)
        X = X.expand(-1, 3, -1, -1)
        Y = Y.expand(-1, 3, -1, -1)
        xx = self.slice1(X)
        fx2 = xx
        xx = self.slice2(xx)
        fx4 = xx
        xx = self.slice3(xx)
        fx6 = xx

        yy = self.slice1(Y)
        fy2 = yy
        yy = self.slice2(yy)
        fy4 = yy
        yy = self.slice3(yy)
        fy6 = yy
        loss_p = self.maeloss(fx2, fy2) + self.maeloss(fx4, fy4) + self.maeloss(fx6, fy6)

        return loss_p


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)
        return E


class frequency(nn.Module):

    def __init__(self):
        super(frequency, self).__init__()

    def forward(self, org, enhance):
        return E


# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class frequency(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False,
                 batch_matrix=False):
        super(frequency, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def gaussian_filter(self, h, w, sigma, mu):
        x, y = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
        d = np.sqrt(x * x + y * y)
        sigma_, mu_ = sigma, mu  # 0.5, 0
        g = np.exp(-((d - mu_) ** 2 / (2.0 * sigma_ ** 2)))

        return g

    def loss_formulation(self, recon_freq, real_freq1, real_freq2, matrix=None, gaussian1_=None, gaussian2_=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            gaussian_1 = gaussian1_.view(1, recon_freq.size(1), recon_freq.size(2), recon_freq.size(3),
                                         recon_freq.size(4))
            gaussian_2 = gaussian2_.view(1, recon_freq.size(1), recon_freq.size(2), recon_freq.size(3),
                                         recon_freq.size(4))

            new_recon_freq1 = recon_freq
            new_recon_freq1[:, :, :, :, :, 0] = gaussian_1 * new_recon_freq1[:, :, :, :, :, 0]
            new_recon_freq2 = recon_freq
            new_recon_freq2[:, :, :, :, :, 0] = gaussian_2 * new_recon_freq2[:, :, :, :, :, 0]
            new_real_freq1 = real_freq1
            new_real_freq1[:, :, :, :, :, 0] = gaussian_1 * real_freq1[:, :, :, :, :, 0]
            new_real_freq2 = real_freq2
            new_real_freq2[:, :, :, :, :, 0] = gaussian_2 * real_freq2[:, :, :, :, :, 0]

            matrix_tmp1 = (new_recon_freq1 - new_real_freq1) ** 2
            matrix_tmp1 = torch.sqrt(matrix_tmp1[..., 0] + matrix_tmp1[..., 1]) ** self.alpha
            matrix_tmp2 = (new_recon_freq2 - new_real_freq2) ** 2
            matrix_tmp2 = torch.sqrt(matrix_tmp2[..., 0] + matrix_tmp2[..., 1]) ** self.alpha
            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp1 = torch.log(matrix_tmp1 + 1.0)
                matrix_tmp2 = torch.log(matrix_tmp2 + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp1 = matrix_tmp1 / matrix_tmp1.max()
                matrix_tmp2 = matrix_tmp2 / matrix_tmp2.max()
            else:
                matrix_tmp1 = matrix_tmp1 / matrix_tmp1.max(-1).values.max(-1).values[:, :, :, None, None]
                matrix_tmp2 = matrix_tmp2 / matrix_tmp2.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp1[torch.isnan(matrix_tmp1)] = 0.0
            matrix_tmp1 = torch.clamp(matrix_tmp1, min=0.0, max=1.0)
            weight_matrix1 = matrix_tmp1.clone().detach()

            matrix_tmp2[torch.isnan(matrix_tmp2)] = 0.0
            matrix_tmp2 = torch.clamp(matrix_tmp2, min=0.0, max=1.0)
            weight_matrix2 = matrix_tmp2.clone().detach()

            new_weight_matrix1 = weight_matrix1 / (weight_matrix1 + weight_matrix2)
            new_weight_matrix2 = weight_matrix2 / (weight_matrix1 + weight_matrix2)

        # frequency distance using (squared) Euclidean distance
        tmp1 = (recon_freq - real_freq1) ** 2
        freq_distance1 = tmp1[..., 0] + tmp1[..., 1]

        tmp2 = (recon_freq - real_freq2) ** 2
        freq_distance2 = tmp2[..., 0] + tmp2[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = new_weight_matrix1 * freq_distance1 + new_weight_matrix2 * freq_distance2
        return torch.mean(loss)

    def forward(self, pred, target1, target2, matrix=None, **kwargs):
        pred_freq = self.tensor2freq(pred)
        target_freq1 = self.tensor2freq(target1)
        target_freq2 = self.tensor2freq(target2)

        gaussian1 = self.gaussian_filter(pred_freq.size(3), pred_freq.size(4), 2.0, 0)
        gaussian1_ = fft.ifftshift(gaussian1)
        gaussian1_ = torch.Tensor(gaussian1_).cuda()

        gaussian2 = self.gaussian_filter(pred_freq.size(3), pred_freq.size(4), 1.0, 0)
        gaussian2_ = fft.ifftshift(gaussian2)
        gaussian2_ = torch.Tensor(gaussian2_).cuda()
        
        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq1 = torch.mean(target_freq1, 0, keepdim=True)
            target_freq2 = torch.mean(target_freq2, 0, keepdim=True)

        # calculate focal frequency loss
        loss_frequency = self.loss_formulation(pred_freq, target_freq1, target_freq2, matrix, gaussian1_, gaussian2_)
        return loss_frequency
