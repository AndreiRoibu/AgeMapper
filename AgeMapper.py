from torch import prod, tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgeMapper(nn.Module):
    """
    AgeMapper is a 3D Convolutional Neural Network (CNN) that maps a 3D brain image to a single age value.
    The network is composed of a feature extractor and a fully connected layer.
    The feature extractor is composed of 5 convolutional layers with max pooling and ReLU activation.
    The fully connected layer is composed of 3 fully connected layers with ReLU activation.
    The network is trained using the mean squared error loss function.
    """
    def __init__(self, 
                resolution: str = '1mm', 
                channel_number: list = [32,64,64,64,64],
                dropout_rate_1: int = 0, 
                dropout_rate_2: int = 0,
                dropout_rate_3: int = 0,
                ) -> None:
        """
        Parameters:
        -----------
        resolution: str
            The resolution of the input image. It can be either '1mm' or '2mm'.
        channel_number: list
            The number of channels in each convolutional layer.
        dropout_rate_1: int
            The dropout rate of the first fully connected layer.
        dropout_rate_2: int
            The dropout rate of the second fully connected layer.
        dropout_rate_3: int 
            The dropout rate of the third fully connected layer.

        Returns:
        --------
        None
        """
        super(AgeMapper, self).__init__()

        # Check if the resolution is supported
        number_of_layers = len(channel_number)

        if resolution=='2mm':
            self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        elif resolution=='1mm':
            self.Upsample = nn.Identity()
        else:
            print("ATTENTION! Resolution >>{}<< Not Supported!!!".format(resolution))


        # Construct the network. The feature extractor is composed of 5 convolutional layers with max pooling and ReLU activation.
        # The fully connected layer is composed of 3 fully connected layers with ReLU activation.

        self.Feature_Extractor = nn.Sequential()
        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = 1
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]

            self.Feature_Extractor.add_module(
                name = 'Convolution_%d' % layer_number,
                module = self._convolutional_block(
                    input_channels,
                    output_channels,
                    maxpool_flag = True,
                    kernel_size = 3,
                    padding_flag= True
                )
            )

        self.FullyConnected = nn.Sequential()
        input_dimensions = 5 * 6 * 5 * output_channels

        if dropout_rate_1 > 0:
            self.FullyConnected.add_module(
                name='Dropout_FullyConnected_3',
                module=nn.Dropout(dropout_rate_1)
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_3',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_3',
            module= nn.ReLU()
        )

        if dropout_rate_2 > 0:
            self.FullyConnected.add_module(
                name='Dropout_FullyConnected_2',
                module=nn.Dropout(dropout_rate_2)
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )
        self.FullyConnected.add_module(
            name = 'ReluActivation_2',
            module= nn.ReLU()
        )

        if dropout_rate_3 > 0:
            self.FullyConnected.add_module(
                name='Dropout_FullyConnected_1',
                module=nn.Dropout(dropout_rate_3)
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module= nn.Linear(
                in_features=32,
                out_features=1,
            )
        )
        self.FullyConnected.add_module(
            name = 'LinearActivation',
            module= nn.Identity()
        )

    # This is a static method. It does not require the class to be instantiated. It is used to define the convolutional blocks. 
    @staticmethod 
    def _convolutional_block(input_channels: int, 
                             output_channels: int, 
                             maxpool_flag: bool = True, 
                             kernel_size: int = 3, 
                             padding_flag: bool = True, 
                             maxpool_stride: int = 2) -> nn.Sequential:
        """
        Static method that defines a convolutional block.

        Parameters:
        -----------
        input_channels: int
            The number of input channels.
        output_channels: int
            The number of output channels.
        maxpool_flag: bool
            If True, a max pooling layer is added to the block.
        kernel_size: int
            The kernel size of the convolutional layer.
        padding_flag: bool
            If True, the convolutional layer is padded.
        maxpool_stride: int
            The stride of the max pooling layer.

        Returns:
        --------   
        layer: nn.Sequential
            The convolutional block.

        """
        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                nn.ReLU()
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.ReLU()
            )

        return layer

    def forward(self, 
                X: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass of the network.

        Parameters:
        -----------
        X: torch.Tensor
            The input image.

        Returns:
        --------
        X: torch.Tensor
            The predicted age.

        """

        X = self.Upsample(X)
        X = self.Feature_Extractor(X)
        X = X.reshape(-1, prod(tensor(X.shape)[1:]))
        X = self.FullyConnected(X)
        return X

class SFCN(nn.Module):
    """
    SFCN is a 3D Convolutional Neural Network (CNN) that maps a 3D brain image to a single age value.
    The architecture corresponds to the SFCN network proposed by Han et al. (2021).
    The publication can be accessed here: Accurate brain age prediction with lightweight deep neural networks <https://www.sciencedirect.com/science/article/pii/S1361841520302358?via%3Dihub>
    """

    def __init__(self, 
                 channel_number: list = [32,64,128,256,256,64], 
                 output_dimension: int = 40, 
                 dropout_flag: bool = True) -> None:
        """
        Parameters:
        -----------
        channel_number: list
            The number of channels in each convolutional layer.
        output_dimension: int
            The number of output channels.
        dropout_flag: bool
            If True, a dropout layer is added to the network.

        Returns:
        --------
        None
        """
        super(SFCN, self).__init__()
        number_of_layers = len(channel_number)

        self.Feature_Extractor = nn.Sequential()
        for layer_number in range(number_of_layers):      
            if layer_number == 0:
                input_channels = 1
            else:
                input_channels = channel_number[layer_number - 1]
            output_channels = channel_number[layer_number]
            if layer_number < number_of_layers-1:
                self.Feature_Extractor.add_module(
                    name = 'Convolution_%d' % layer_number,
                    module = self._convolutional_block(
                        input_channels,
                        output_channels,
                        maxpool_flag = True,
                        kernel_size = 3,
                        padding_flag= True
                    )
                )
            else:
                self.Feature_Extractor.add_module(
                    name = 'Convolution_%d' % layer_number,
                    module = self._convolutional_block(
                        input_channels,
                        output_channels,
                        maxpool_flag = False,
                        kernel_size = 1,
                        padding_flag= False
                    )
                )

        self.Classifier = nn.Sequential()
        output_shape = [5,6,5]
        self.Classifier.add_module(
            name = "Average_Pool",
            module = nn.AvgPool3d(output_shape)
        )

        if dropout_flag == True:
            self.Classifier.add_module('Dropout', nn.Dropout(0.5))
            
        input_channels = channel_number[-1]
        output_channels = output_dimension
        layer_number = number_of_layers
        self.Classifier.add_module(
            name = 'Convolution_%d' % layer_number,
            module = nn.Conv3d(
                in_channels=input_channels,
                out_channels=output_channels,
                padding=0,
                kernel_size=1,
                bias=True
            )
        )

    @staticmethod
    def _convolutional_block(input_channels: int, output_channels: int, maxpool_flag=True, kernel_size=3, padding_flag=True, maxpool_stride=2) -> nn.Sequential: 
        """
        Static method that defines a convolutional block.

        Parameters:
        -----------
        input_channels: int
            The number of input channels.
        output_channels: int
            The number of output channels.  
        maxpool_flag: bool
            If True, a max pooling layer is added to the block. 
        kernel_size: int
            The kernel size of the convolutional layer. 
        padding_flag: bool
            If True, the convolutional layer is padded.
        maxpool_stride: int
            The stride of the max pooling layer.

        Returns:
        --------
        layer: nn.Sequential
            The convolutional block.

        """
                
        if padding_flag == True:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = 0
        
        if maxpool_flag is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=maxpool_stride
                ),
                nn.ReLU()
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.BatchNorm3d(
                    num_features=output_channels,
                    affine=True
                ),
                nn.ReLU()
            )

        return layer

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters:
        -----------
        X: torch.Tensor
            The input image.

        Returns:
        --------
        X: torch.Tensor
            The predicted age.

        """
        
        X = self.Feature_Extractor(X)
        X = self.Classifier(X)
        X = F.log_softmax(X, dim=1)
        return X