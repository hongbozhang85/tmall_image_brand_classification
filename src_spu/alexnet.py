"""
requires pytorch 0.4.0+

Alexnet

torchivision.models.alexnet doesn't include LRN,
we add LRN and BN here.
Furthermore, we add some hyperparameters here for future comparison.

Hongbo
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# dropoutParam = 0.5
# inputSize = 224
    # input image size: inputSize * inputSize
    # if inputPixel changes, 
    # Linear() in self.ann1 will be change as well
    # self.forward will be changed as well.
# num_hidden_1 = 128
# num_hidden_2 = 128
# num_class = 4

class AlexNet(nn.Module):
    def __init__(self, inputSize=224, num_class=4, num_hidden_1=128, num_hidden_2=128, dropoutParam=0.5):
        super(AlexNet, self).__init__()
        self.inputSize = inputSize
        self.num_class = num_class
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.dropoutParam = dropoutParam
        #self.convSize = int((inputSize - 11 + 4)/4 + 1)
        self.convSize = int((self.inputSize - 224)/32 + 6)
        tmp = int((self.inputSize - 224)/32 + 6)

        # input 224*224*3 
        # after conv 55*55*96 
        # after pool 27*27*96
        self.feature1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96 , kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96, momentum=0.5),
            nn.ReLU(),
            nn.LocalResponseNorm(size= 5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2) 
        )
        # input 27*27*96
        # after conv 27*27*256
        # after pool 13*13*256
        self.feature2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256 , kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256, momentum=0.5),
            nn.ReLU(),
            nn.LocalResponseNorm(size= 5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2) 
        )
        # input 13*13*256
        # after cov 13*13*384
        self.feature3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384, momentum=0.5),
            nn.ReLU(),
            # nn.LocalResponseNorm(size= 5, alpha=0.0001, beta=0.75, k=2),
        )
        # input 13*13*384
        # after cov 13*13*384
        self.feature4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384, momentum=0.5),
            nn.ReLU(),
            # nn.LocalResponseNorm(size= 5, alpha=0.0001, beta=0.75, k=2),
        )
        # input 13*13*384
        # after conv 13*13*256
        # after pool 6*6*256
        self.feature5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256 , kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.5),
            nn.ReLU(),
            # nn.LocalResponseNorm(size= 5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2) 
        )
        # input 6*6*256
        # output 4096
        self.ann1 = nn.Sequential(
            nn.Dropout(self.dropoutParam),
            # nn.Linear(256*6*6, self.num_hidden_1),
            nn.Linear(256*tmp*tmp, self.num_hidden_1),
            nn.BatchNorm1d(num_hidden_1, momentum=0.5),
            nn.ReLU(),
        )
        # input 4096
        # output 4096
        self.ann2 = nn.Sequential(
            nn.Dropout(self.dropoutParam),
            nn.Linear(self.num_hidden_1, self.num_hidden_2),
            nn.BatchNorm1d(num_hidden_2, momentum=0.5),
            nn.ReLU(),
        )
        # input 4096
        # output 100
        self.out = nn.Sequential(
            nn.Linear(self.num_hidden_2, self.num_class)
        )
        
    def forward(self,x):
        # print("[" + str(np.size(x, 0))+ " , " + str(np.size(x, 1))  + " , " + str(np.size(x, 2)) + " , " + str(np.size(x, 3)) + "]")
        f1out = self.feature1(x.float())
        f2out = self.feature2(f1out) 
        f3out = self.feature3(f2out)
        f4out = self.feature4(f3out)
        f5out = self.feature5(f4out)
        # fout = f5out.view(f5out.size(0), 256*6*6)
        fout = f5out.view(f5out.size(0), 256*self.convSize*self.convSize)
        a1out = self.ann1(fout)
        a2out = self.ann2(a1out)
        out   = self.out(a2out)
        return out
        
