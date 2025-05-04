namespace DotML.Network;

/// <summary>
/// Utility methods related to the UNet Convolutional Neural Network architecture
/// <see href="https://en.wikipedia.org/wiki/U-Net"/>
/// </summary>
public static class UNet {

    //https://medium.com/analytics-vidhya/what-is-unet-157314c87634

    public static FeedforwardNetwork Make(int img_channels, int img_width, int img_height, int output_channels, int depth, ActivationFunction? activation = null, bool normalize = false, float? dropout = null) {
        depth        = Math.Max(1, depth);
        img_channels = Math.Max(1, img_channels);
        output_channels = Math.Max(1, output_channels);
        img_width    = Math.Max(1, img_width);
        img_height   = Math.Max(1, img_height);
        activation   = activation ?? ReLU.Instance;

        var network = new FeedforwardNetwork(
            // The 'U' shape
            level(0, depth, new Shape3D(img_channels, img_height, img_width), activation, normalize, dropout)
            // The output segments(s)
            .Then((ishape) => new ConvolutionLayer(
                input_size: ishape,
                padding: Padding.Same,
                filters: ConvolutionFilter.Make(filters: output_channels, kernels_per_filter: ishape.Channels, 1)
            ))
        );
        network.Name = $"UNet-depth-{depth}";

        return network;
    }

    private static IEnumerable<IFeedforwardNetworkLayer> level(int current_depth, int max_depth, Shape3D input_shape, ActivationFunction activation, bool normalize, float? dropout) {
        if (current_depth > max_depth || current_depth < 0)
            yield break;

        var num_features = Math.Max(input_shape.Channels * 2, 64); // At least 64 features

        // TODO maybe add DROPOUT after relu's

        if (current_depth == max_depth) {
            // TODO bridge
            var bconv_1 = new ConvolutionLayer(
                input_size: input_shape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: num_features, kernels_per_filter: input_shape.Channels, 3)
            );
            yield return bconv_1;
            if (normalize) {
                yield return new LayerNorm(bconv_1.OutputShape);
            }
            var bact_1 = new ActivationLayer(input_size: bconv_1.OutputShape, activation);
            yield return bact_1;

            var bconv_2 = new ConvolutionLayer(
                input_size: bact_1.OutputShape,
                padding: Padding.Same,
                stride: 1,
                filters: ConvolutionFilter.Make(filters: num_features, kernels_per_filter: bact_1.OutputShape.Channels, 3)
            );
            yield return bconv_2;
            if (normalize) {
                yield return new LayerNorm(bconv_2.OutputShape);
            }
            var bact_2 = new ActivationLayer(input_size: bconv_2.OutputShape, activation);
            yield return bact_2; 

            // DONE WITH RECURSION!
            yield break;
        }

        // Encoder path
        var down_conv_1 = new ConvolutionLayer(
            input_size: input_shape,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(filters: num_features, kernels_per_filter: input_shape.Channels, 3)
        );
        yield return down_conv_1;
        if (normalize) {
            yield return new LayerNorm(down_conv_1.OutputShape);
        }
        var act_1 = new ActivationLayer(input_size: down_conv_1.OutputShape, activation);
        yield return act_1;

        var down_conv_2 = new ConvolutionLayer(
            input_size: act_1.OutputShape,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(filters: num_features, kernels_per_filter: act_1.OutputShape.Channels, 3)
        );
        yield return down_conv_2;
        if (normalize) {
            yield return new LayerNorm(down_conv_2.OutputShape);
        }
        var act_2 = new ActivationLayer(input_size: down_conv_2.OutputShape, activation);
        yield return act_2; 

        var capture = new InputCapture(act_2.OutputShape);
        yield return capture;

        // Descend to lower layers
        var pool = new LocalMaxPoolingLayer(input_size: capture.OutputShape, 3);
        foreach (var layer in level(current_depth + 1, max_depth, pool.OutputShape, activation, normalize, dropout)) {
            yield return layer;
        }
        var up_conv = new TransposeConvolutionLayer(
            input_size: pool.OutputShape, 
            padding: Padding.Same,
            expansion: Expansion.Expand,
            strideX: 1, 
            strideY: 1,
            filters: ConvolutionFilter.Make(filters: pool.OutputShape.Channels, kernels_per_filter: pool.OutputShape.Channels, 3)
        );
        yield return up_conv;

        // Decoder path
        var skip = new ConcatenationSkipConnection(up_conv.OutputShape, capture, ConcatenationSkipConnection.Side.ResidualLeft);
        yield return skip;

        var compress_1 = new ConvolutionLayer(
            input_size: skip.OutputShape,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(filters: num_features, kernels_per_filter: skip.OutputShape.Channels, 3)
        );
        yield return compress_1;
        if (normalize) {
            yield return new LayerNorm(compress_1.OutputShape);
        }
        var act_3 = new ActivationLayer(input_size: compress_1.OutputShape, activation);
        yield return act_3; 

        var compress_2 = new ConvolutionLayer(
            input_size: act_3.OutputShape,
            padding: Padding.Same,
            stride: 1,
            filters: ConvolutionFilter.Make(filters: num_features, kernels_per_filter: act_3.OutputShape.Channels, 3)
        );
        yield return compress_2;
        if (normalize) {
            yield return new LayerNorm(compress_2.OutputShape);
        }
        var act_4 = new ActivationLayer(input_size: compress_2.OutputShape, activation);
        yield return act_4; 
    }
}