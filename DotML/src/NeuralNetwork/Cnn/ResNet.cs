namespace DotML.Network;

/// <summary>
/// Utility methods related to the ResNet (Residual Network) architecture
/// <see href="https://en.wikipedia.org/wiki/Residual_neural_network"/>
/// </summary>
public static class ResNet {
    /// <summary>
    /// Supported ResNet versions
    /// </summary>
    public enum Version {
        V6 = 6,
        V18 = 18,
        /// <summary>
        /// Latest supported version
        /// </summary>
        Latest = 18
    }

    /// <summary>
    /// Typical number of channels per image processed by ResNet (typically three, RGB)
    /// </summary>
    public const int IMG_CHANNELS = 3;
    /// <summary>
    /// Typical width (in pixels) for an image processed by ResNet (typically 224 pixels)
    /// </summary>
    public const int IMG_WIDTH = 224;
    /// <summary>
    /// Typical height (in pixels) for an image processed by ResNet (typically 224 pixels)
    /// </summary>
    public const int IMG_HEIGHT = 224;
    /// <summary>
    /// Typical number of classes for an image classification task (typically 1000 classes for ImageNet)    
    /// </summary>
    public const int OUTPUT_CLASSES = 1000;

    public static FeedforwardNetwork Make(Version version, int img_channels = IMG_CHANNELS, int img_width = IMG_WIDTH, int img_height = IMG_HEIGHT, int output_classes = OUTPUT_CLASSES, ActivationFunction? activation = null) {
        var net = version switch {
            Version.V6 => MakeV6(img_channels, img_width, img_height, output_classes, activation),
            Version.V18 => MakeV18(output_classes, img_channels, img_width, img_height, activation),
            _ => throw new ArgumentException(nameof(version))
        };
        net.Name = "ResNet-" + ((int)version);

        return net;
    }

    private static IEnumerable<IFeedforwardNetworkLayer> V6Block(Shape3D size, ActivationFunction activation) {
        var capture = new InputCapture(size);
        yield return capture;

        var conv1 = new ConvolutionLayer(
            input_size: size,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(filters: size.Channels, kernels_per_filter: size.Channels, kernel_size: 3)
        );
        yield return conv1;

        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return norm1;

        var active1 = new ActivationLayer(norm1.OutputShape, activation);
        yield return active1;

        // ---

        var conv2 = new ConvolutionLayer(
            input_size: active1.OutputShape,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(filters: size.Channels, kernels_per_filter: size.Channels, kernel_size: 3)
        );
        yield return conv2;

        var norm2 = new LayerNorm(conv2.OutputShape);
        yield return norm2;

        var residual = new AdditionSkipConnection(norm2.OutputShape, capture);
        yield return residual;

        yield return new ActivationLayer(residual.OutputShape, activation);
    }

    private static FeedforwardNetwork MakeV6(int img_channels, int img_width, int img_height, int output_classes, ActivationFunction? activation) {
        activation = activation ?? ReLU.Instance;

        return new FeedforwardNetwork(
            new ConvolutionLayer(
                input_size: new Shape3D(img_channels, img_height, img_width),
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: 16, kernels_per_filter: img_channels, kernel_size: 3)
            )
            .Then((ishape) => new LayerNorm(ishape))
            .WithActivation(activation)
            .Then((ishape) => V6Block(ishape, activation))
            .Then((ishape) => V6Block(ishape, activation))
            .Then((ishape) => new LocalAvgPoolingLayer(ishape, size: 3))
            .Then((ishape) => new DenseLinearLayer(ishape.Count, output_classes))
            .Then((ishape) => new SoftmaxLayer(ishape.Count))
        );    
    }

    // https://medium.com/analytics-vidhya/resnet-understand-and-implement-from-scratch-d0eb9725e0db
    private static IEnumerable<IFeedforwardNetworkLayer> V18StartBlock(Shape3D input_shape, ActivationFunction activation) {
        // input=(3x224x224) -> output=(64x56x56)
        var conv1 = new ConvolutionLayer(
            input_size: input_shape,
            padding: Padding.Same,
            strideX: 2,
            strideY: 2,
            filters: ConvolutionFilter.Make(64, input_shape.Channels, 7)
        );
        yield return conv1;

        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return conv1;

        var act1 = new ActivationLayer(norm1.OutputShape, activation);
        yield return act1;

        var pool1 = new LocalMaxPoolingLayer(conv1.InputShape, size: 3, stride: 2, padding: 1); // TODO add padding
        yield return pool1;
    }

    private static IEnumerable<IFeedforwardNetworkLayer> V18ResidualBlock2(Shape3D input_shape, ActivationFunction func) {
        // input=(64x56x56) -> output = (64x56x56)
        var capture = new InputCapture(input_shape);
        yield return capture;

        // Conv1
        var conv1 = new ConvolutionLayer(
            input_size: input_shape,
            padding: Padding.Valid,
            stride: 1, 
            filters: ConvolutionFilter.Make(64, input_shape.Channels, 3)
        );
        yield return conv1;

        // Norm1
        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return norm1;

        // Activation1
        var act1 = new ActivationLayer(norm1.OutputShape, func);
        yield return act1;

        // Conv2
        var conv2 = new ConvolutionLayer(
            input_size: act1.OutputShape,
            padding: Padding.Valid,
            stride: 1, 
            filters: ConvolutionFilter.Make(64, act1.OutputShape.Channels, 3)
        );
        yield return conv2;

        // Norm2
        var norm2 = new LayerNorm(conv2.OutputShape);
        yield return norm2;

        // TODO add dropout?

        // The skip connection endpoint
        var skip = new AdditionSkipConnection(norm2.OutputShape, capture);
        yield return skip;

        // Activation2
        var act2 = new ActivationLayer(skip.OutputShape, func);
        yield return act2;
    }

    private static IEnumerable<IFeedforwardNetworkLayer> V18ResidualBlock3A(Shape3D input, ActivationFunction func) {
        var capture = new InputCapture(input);
        yield return capture;

        var conv1 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(128, input.Channels, 3)
        );
        yield return conv1;

        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return norm1;

        var act2 = new ActivationLayer(norm1.OutputShape, func);
        yield return act2;

        var conv2 = new ConvolutionLayer(
            input_size: act2.OutputShape, 
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(128, act2.OutputShape.Channels, 3)
        );
        yield return conv2;

        var norm2 = new LayerNorm(conv2.OutputShape);
        yield return norm2;

        // The skip connection
        var squeeze = new ConvolutionLayer(
            input_size: norm2.OutputShape,
            padding: Padding.Valid,
            stride: 2,
            filters: ConvolutionFilter.Make(capture.OutputShape.Channels, norm2.OutputShape.Channels, 1)
        );
        yield return squeeze;

        var skip = new AdditionSkipConnection(squeeze.OutputShape, capture);
        yield return skip;

        yield return new ActivationLayer(skip.OutputShape, func);
    }

    private static IEnumerable<IFeedforwardNetworkLayer> V18ResidualBlock3B(Shape3D input, ActivationFunction func) {
        var capture = new InputCapture(input);
        yield return capture;

        var conv1 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(input.Count, input.Count, 3)
        );
        yield return conv1;

        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return norm1;

        var act1 = new ActivationLayer(norm1.OutputShape, func);
        yield return act1;

        var conv2 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(input.Count, input.Count, 3)
        );
        yield return conv2;

        var norm2 = new LayerNorm(conv2.OutputShape);
        yield return norm2;

        // TODO add dropout?

        // The skip connection
        var skip = new AdditionSkipConnection(norm2.OutputShape, capture);
        yield return skip;

        var act2 = new ActivationLayer(skip.OutputShape, func);
        yield return act2;
    }

    private static IEnumerable<IFeedforwardNetworkLayer> V18ResidualBlock4A(Shape3D input, ActivationFunction func) {
        var capture = new InputCapture(input);
        yield return capture; // Capture the input from Block3

        var conv1 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 2,
            filters: ConvolutionFilter.Make(256, input.Channels, 3)
        );
        yield return conv1;

        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return norm1;

        var act2 = new ActivationLayer(norm1.OutputShape, func);
        yield return act2;

        var conv2 = new ConvolutionLayer(
            input_size: act2.OutputShape, 
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(act2.OutputShape.Channels, act2.OutputShape.Channels, 3)
        );
        yield return conv2;

        var norm2 = new LayerNorm(conv2.OutputShape);
        yield return norm2;

        // TODO add dropout?

        // The skip connection
        var squeeze = new ConvolutionLayer(
            input_size: norm2.OutputShape,
            padding: Padding.Valid,
            stride: 2,
            filters: ConvolutionFilter.Make(capture.OutputShape.Channels, norm2.OutputShape.Channels, 1)
        );
        yield return squeeze;

        var skip = new AdditionSkipConnection(squeeze.OutputShape, capture);
        yield return skip;

        yield return new ActivationLayer(skip.OutputShape, func);
    }

    private static IEnumerable<IFeedforwardNetworkLayer> V18ResidualBlock4B(Shape3D input, ActivationFunction func) {
        var capture = new InputCapture(input);
        yield return capture; // capture the input from the first part of block 4

        var conv1 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(input.Count, input.Count, 3)
        );
        yield return conv1;

        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return norm1;

        var act1 = new ActivationLayer(norm1.OutputShape, func);
        yield return act1;

        var conv2 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(input.Count, input.Count, 3)
        );
        yield return conv2;

        var norm2 = new LayerNorm(conv2.OutputShape);
        yield return norm2;

        // TODO add dropout?

        // The skip connection
        var skip = new AdditionSkipConnection(norm2.OutputShape, capture);
        yield return skip;

        var act2 = new ActivationLayer(skip.OutputShape, func);
        yield return act2;
    }

    private static IEnumerable<IFeedforwardNetworkLayer> V18ResidualBlock5A(Shape3D input, ActivationFunction func) {
        var capture = new InputCapture(input);
        yield return capture; // Capture the input from Block 4

        var conv1 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 2,
            filters: ConvolutionFilter.Make(512, input.Channels, 3)
        );
        yield return conv1;

        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return norm1;

        var act2 = new ActivationLayer(norm1.OutputShape, func);
        yield return act2;

        var conv2 = new ConvolutionLayer(
            input_size: act2.OutputShape, 
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(act2.OutputShape.Channels, act2.OutputShape.Channels, 3)
        );
        yield return conv2;

        var norm2 = new LayerNorm(conv2.OutputShape);
        yield return norm2;

        // TODO add dropout?

        // The skip connection
        var squeeze = new ConvolutionLayer(
            input_size: norm2.OutputShape,
            padding: Padding.Valid,
            stride: 2,
            filters: ConvolutionFilter.Make(capture.OutputShape.Channels, norm2.OutputShape.Channels, 1)
        );
        yield return squeeze;

        var skip = new AdditionSkipConnection(squeeze.OutputShape, capture);
        yield return skip;

        yield return new ActivationLayer(skip.OutputShape, func);
    }

    private static IEnumerable<IFeedforwardNetworkLayer> V18ResidualBlock5B(Shape3D input, ActivationFunction func) {
        var capture = new InputCapture(input);
        yield return capture; // capture the input from the first part of block 4

        var conv1 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(input.Count, input.Count, 3)
        );
        yield return conv1;

        var norm1 = new LayerNorm(conv1.OutputShape);
        yield return norm1;

        var act1 = new ActivationLayer(norm1.OutputShape, func);
        yield return act1;

        var conv2 = new ConvolutionLayer(
            input_size: input,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(input.Count, input.Count, 3)
        );
        yield return conv2;

        var norm2 = new LayerNorm(conv2.OutputShape);
        yield return norm2;

        // TODO add dropout?

        // The skip connection
        var skip = new AdditionSkipConnection(norm2.OutputShape, capture);
        yield return skip;

        var act2 = new ActivationLayer(skip.OutputShape, func);
        yield return act2;
    }
    private static IEnumerable<IFeedforwardNetworkLayer> V18ClassificationBlock(Shape3D input_shape, int output_classes, ActivationFunction func) {
        var av1 = new LocalAvgPoolingLayer(input_shape, stride: 1, size: 7);
        yield return av1;

        var fc1 = new DenseLinearLayer(av1.OutputShape.Count, 1000);
        yield return fc1;

        var act1 = new ActivationLayer(fc1.OutputShape, func);
        yield return act1;

        var fc2 = new DenseLinearLayer(fc1.OutputShape.Count, output_classes);
        yield return fc2;

        var softmax = new SoftmaxLayer(fc2.OutputShape.Count);
        yield return softmax;
    }

    private static FeedforwardNetwork MakeV18(int output_classes, int img_channels, int img_width, int img_height, ActivationFunction? activation) {
        activation = activation ?? ReLU.Instance;

        return new FeedforwardNetwork(
            // Staring Block
            V18StartBlock(new Shape3D(img_channels, img_height, img_width), activation)
            // Block 2 (repeated 2x)
            .Then((ishape) => V18ResidualBlock2(ishape, activation))
            .Then((ishape) => V18ResidualBlock2(ishape, activation))
            // Block 3
            .Then((ishape) => V18ResidualBlock3A(ishape, activation))
            .Then((ishape) => V18ResidualBlock3B(ishape, activation))
            // Block 4
            .Then((ishape) => V18ResidualBlock4A(ishape, activation))
            .Then((ishape) => V18ResidualBlock4B(ishape, activation))
            // Block 5 
            .Then((ishape) => V18ResidualBlock5A(ishape, activation))
            .Then((ishape) => V18ResidualBlock5B(ishape, activation))
            // Classification Block
            .Then((ishape) => V18ClassificationBlock(ishape, output_classes, activation))

        );
    }

}

/*
# --- Basic Residual Block with LayerNorm ---
class ResidualBlock(nn.Module):
    def __init__(self, channels, height, width):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm([channels, height, width])
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm([channels, height, width])

    def forward(self, x):
        identity = x
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += identity  # Residual connection
        return F.relu(out)

# --- ResNet-6 Model with LayerNorm and AvgPool2d ---
class ResNet6_LN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet6_LN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm([16, 32, 32])  # Matches CIFAR-10 input size

        # Residual blocks with known spatial dims
        self.block1 = ResidualBlock(16, 32, 32)
        self.block2 = ResidualBlock(16, 32, 32)

        # Fixed-size avg pooling (32 → 1 with kernel 32)
        self.avg_pool = nn.AvgPool2d(kernel_size=32)  # Equivalent to AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        out = F.relu(self.ln(self.conv(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.avg_pool(out)  # Output: (B, 16, 1, 1)
        out = out.view(out.size(0), -1)  # Flatten to (B, 16)
        out = self.fc(out)
        return out
*/