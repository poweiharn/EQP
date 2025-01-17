import torch
import torch.nn as nn
import torch.nn.functional as F

class mixedPool(nn.Module):
    def __init__(self,kernel_size, stride, padding=0, alpha=0.5):
        # nn.Module.__init__(self)
        super(mixedPool, self).__init__()
        if torch.cuda.is_available():
            alpha = torch.FloatTensor([alpha]).cuda()
        else:
            alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)  # nn.Parameter is special Variable
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding) + (
                    1 - self.alpha) * F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x

def X():
    #print("You are using Mixed Pooling Method")
    return mixedPool