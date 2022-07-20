from torch import nn

class Net(nn.Module):
    
    def __init__(self, n_classes=10):
        super().__init__()
        
        
        # Cheatsheet:
        # Convolutions (Conv2d) need to match the input's number of channels, and can change the number of output channels
        # It's recommended that you stick to 3x3 convolutions with padding=1, which keeps the spatial size of the tensors unchanged
        # Batch normalization (BatchNorm2d) should match the input's number of channels, and do not affect the size
        # BatchNorm2d is used right after Conv2d
        # ReLU activation can be applied to tensors of any size and does not affect its shape
        # Max pooling (MaxPool2d) divides the spatial dimensions of the image. 
        # Stick to MaxPool2d(2, 2) in this exercise, which divides the size of the output by 2 in each axis.

        # Note: Tensors (arrays) here are of shape [B, C, M, N], where:
        # B is the number of samples in a batch
        # C is the number of "color" or feature channels
        # M is the first spatial dimension
        # N is the second spatial dimension

        # Tip: try to keep track of the shape that the tensor will have after each operation. This is necessary for properly 
        # connecting the convolutional layers to the fully connected layers.

        # Input is [B, 3, 64, 64]
        self.convolutions = nn.Sequential( 
            nn.Conv2d(3, 4, 3, padding=1), # [B, 4, 64, 64]
            nn.ReLU(inplace=True), # [B, 4, 64, 64]
            nn.MaxPool2d(2, 2), # [B, 4, 32, 32]
            #==============================================================
            nn.Conv2d(4, 8, 3, padding=1), # [B, 8, 32, 32]
            nn.ReLU(inplace=True), # [B, 8, 32, 32]
            nn.MaxPool2d(2, 2), # [B, 8, 16, 16]
        )

        # Input will be reshaped from [B, 8, 16, 16] to [B, 8*16*16] for fully connected layers
        self.fully_connected = nn.Sequential(
            nn.Linear(8*16*16, 32), # [B, 32]
            nn.ReLU(inplace=True), # [B, 32]
            nn.Linear(32, n_classes), # [B, n_classes]
        )

        # Note: the final output must have shape [B, n_classes]

        # We're skipping a softmax activation here since we'll be using a loss function that does it automatically



    def forward(self, img):

        # Apply convolution operations
        x = self.convolutions(img)

        # Reshape
        x = x.view(x.size(0), -1)

        # Apply fully connected operations
        x = self.fully_connected(x)

        return x