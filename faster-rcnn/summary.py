#--------------------------------------------#
#   This part of the code is used to see the network structure
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.frcnn import FasterRCNN

if __name__ == "__main__":
    input_shape     = [600, 600]
    num_classes     = 21
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = FasterRCNN(num_classes, backbone = 'vgg').to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2 because the profile does not consider the convolution as two operations, some papers count the convolution as two operations,
    #   multiplication and addition. multiply by 2, some papers only consider the number of multiplication operations, ignoring addition. 
    #   In this case, we do not multiply by 2. This code chooses to multiply by 2, refer to YOLOX.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
