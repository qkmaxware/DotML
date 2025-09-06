using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// A layer that helps super-resolution models implement efficient sub-pixel convolutions.
/// <see href="https://paperswithcode.com/method/pixelshuffle"/>
/// </summary>
public class PixelShuffle : FeedforwardNetworkLayer {
    /// <summary>
    /// Upscaling factor for the output image size
    /// </summary>
    public int UpscalingFactor {get; init;}

    public PixelShuffle(Shape3D input_size, int upscale_factor) {
        this.InputShape = input_size;
        upscale_factor = Math.Max(1, upscale_factor);
        this.UpscalingFactor = upscale_factor;

        var inHeight = input_size.Rows;
        var inWidth = input_size.Columns;
        var inChannels = input_size.Channels;

        var rr = upscale_factor * upscale_factor;
        if (inChannels % rr != 0) {
            throw new ArgumentException("Input channels must be divisible by the square of the upscale factor");
        }

        var outChannels = inChannels / rr;
        var outHeight = inHeight * upscale_factor;
        var outWidth = inWidth * upscale_factor;
        this.OutputShape = new Shape3D(outChannels, outHeight, outWidth);
    }

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> input) {
        var inHeight = InputShape.Rows;
        var inWidth = InputShape.Columns;
        var inChannels = InputShape.Channels;

        var r = this.UpscalingFactor;
        var outChannels = OutputShape.Channels;
        var outHeight = OutputShape.Rows;
        var outWidth = OutputShape.Columns;
        var features = new FeatureSet<float>(this.OutputShape);

        for (var channel = 0; channel < outChannels; channel++) {
            var feature = features[channel];

            for (var row = 0; row < inHeight; row++) {
                for (var col = 0; col < inWidth; col++) {
                    for (var i = 0; i < r; i++) {
                        for (var j = 0; j < r; j++) {
                            int inChannel = channel*r*r + i*r + j;
                            int outRow = row * r + i;
                            int outCol = col * r + j;

                            feature[outRow, outCol] += input[inChannel, row, col];
                        }
                    }
                }
            }
        }

        return features;
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        var upscale_factor = this.UpscalingFactor;
        var upscale_factor2 = upscale_factor * upscale_factor;
        var outChannels = args.dY.Channels;
        var outHeight = args.dY.Rows;
        var outWidth = args.dY.Columns;
        var inChannels = outChannels * upscale_factor * upscale_factor;
        var inHeight = outHeight / upscale_factor;
        var inWidth = outWidth / upscale_factor;
        var in_shape = new Shape4D(args.dY.Batches, inChannels, inHeight, inWidth);
        var dX = new BatchedFeatureSet<float>(in_shape);

        Parallel.For(0, in_shape.Batches, (batch) => {
            var error_features = args.OutputErrors[batch];
            var dx_batch = dX[batch];

            for (var channel = 0; channel < outChannels; channel++) {
                var error_feature = error_features[channel];

                for (var row = 0; row < inHeight; row++) {
                    for (var col = 0; col < inWidth; col++) {
                        for (var i = 0; i < upscale_factor; i++) {
                            for (var j = 0; j < upscale_factor; j++) {
                                int inChannel = channel*upscale_factor2 + i*upscale_factor + j;
                                int outRow = row * upscale_factor + i;
                                int outCol = col * upscale_factor + j;

                                dx_batch[inChannel, row, col] += error_feature[outRow, outCol];
                            }
                        }
                    }
                }
            }
        });
        
        return new BackpropagationReturns(
            error: dX,
            gradients: null
        );
    }

    public override void Initialize(IInitializer initializer) { }

    public override void SubtractGradients(LayerGradients? gradients) { }

    public override int TrainableParameterCount() => 0;

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);

    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);

    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);

    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}