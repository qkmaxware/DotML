using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// A layer that helps super-resolution models implement efficient sub-pixel convolutions.
/// <see href="https://paperswithcode.com/method/pixelshuffle"/>
/// </summary>
public class PixelShuffler : NetworkLayer
{
    /// <summary>
    /// Upscaling factor for the output image size
    /// </summary>
    public int UpscalingFactor { get; init; }

    public PixelShuffler(int upscale)
    {
        this.UpscalingFactor = upscale;
    }

    public override void Initialize(IInitializer initializer) { }

    public override TensorShape ForwardShape(TensorShape input)
    {
        var shape = input;

        int batches = shape.Length(0);
        int inChannels = shape.Length(1);
        int inHeight = shape.Length(2);
        int inWidth = shape.Length(3);

        int r = this.UpscalingFactor;
        int rr = r * r;
        if (inChannels % rr != 0)
            throw new ArgumentException("Input channels must be divisible by the square of the upscale factor");

        int outChannels = inChannels / rr;
        int outHeight = inHeight * r;
        int outWidth = inWidth * r;

        return new TensorShape(batches, outChannels, outHeight, outWidth);
    }

    public override Tensor<float> Forward(Tensor<float> input)
    {
        // Expect input shape: [batch, inChannels, inHeight, inWidth]
        input = input.ReshapeShared(input.Shape.NormalizeRank(4));
        var shape = input.Shape;

        int batches = shape.Length(0), batchStride = shape.Stride(0);
        int inChannels = shape.Length(1), channelStride = shape.Stride(1);
        int inHeight = shape.Length(2), rowStride = shape.Stride(2);
        int inWidth = shape.Length(3), columnStride = shape.Stride(3);

        int r = this.UpscalingFactor;
        int rr = r * r;
        if (inChannels % rr != 0)
            throw new ArgumentException("Input channels must be divisible by the square of the upscale factor");

        int outChannels = inChannels / rr;
        int outHeight = inHeight * r;
        int outWidth = inWidth * r;

        var output = Tensor<float>.Zeros(new TensorShape(batches, outChannels, outHeight, outWidth));

        Parallel.For(0, batches, (batch) =>
        {
            for (var channel = 0; channel < outChannels; channel++)
            {
                for (var row = 0; row < inHeight; row++)
                {
                    for (var col = 0; col < inWidth; col++)
                    {
                        for (var i = 0; i < r; i++)
                        {
                            for (var j = 0; j < r; j++)
                            {
                                int inChannel = channel * rr + i * r + j;
                                int outRow = row * r + i;
                                int outCol = col * r + j;

                                output[batch, channel, outRow, outCol] += input[batch, inChannel, row, col];
                            }
                        }
                    }
                }
            }
        });

        return output;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var r = this.UpscalingFactor;
        var rr = r * r;

        var outShape = dy.Shape;
        var batches = outShape.Length(0);
        var outChannels = outShape.Length(1);
        var outHeight = outShape.Length(2);
        var outWidth = outShape.Length(3);
        var inChannels = outChannels * rr;

        var inHeight = outHeight / r;
        var inWidth = outWidth / r;
        var inShape = new TensorShape(batches, inChannels, inHeight, inWidth);
        var dX = Tensor<float>.Zeros(inShape);

        Parallel.For(0, batches, (batch) =>
        {
            for (var channel = 0; channel < outChannels; channel++)
            {
                for (var row = 0; row < inHeight; row++)
                {
                    for (var col = 0; col < inWidth; col++)
                    {
                        for (var i = 0; i < r; i++)
                        {
                            for (var j = 0; j < r; j++)
                            {
                                int inChannel = channel * rr + i * r + j;
                                int outRow = row * r + i;
                                int outCol = col * r + j;

                                dX[batch, inChannel, row, col] += dy[batch, channel, outRow, outCol];
                            }
                        }
                    }
                }
            }
        });

        return new Gradient(dX);
    }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null) { /* Nothing to do here */ }
    
    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}