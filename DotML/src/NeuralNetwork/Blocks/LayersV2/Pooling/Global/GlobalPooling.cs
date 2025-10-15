using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Apply pooling to reduce the size of the image data
/// <see href="https://en.wikipedia.org/wiki/Pooling_layer"/>
/// </summary>
public abstract class GlobalPooling : Pooling
{

}

/// <summary>
/// Apply pooling to reduce the size of the image data across the last 2 dimensions of the input (Height and Width)
/// <see href="https://en.wikipedia.org/wiki/Pooling_layer"/>
/// </summary>
public abstract class GlobalPooling2D : LocalPooling
{
    public override void Initialize(IInitializer initializer) { }

    // Accumulate value over all spatial positions
    protected abstract float Accumulate(float current, float delta, int count);
    // Final value aggregation (when combined with accumulate it should cover most use cases)
    protected abstract float Aggregate(float current, int count);

    public override TensorShape ForwardShape(TensorShape input)
    {
        var ishape = input.EnsureRank(4); // NCHW
        return ishape.Slice(0..1); // NC
    }

    public override Tensor<float> Forward(Tensor<float> inputs)
    {
        var reshaped = inputs.ReshapeShared(inputs.Shape.EnsureRank(4)); // At least NCHW
        var batches = reshaped.Shape[0];
        var channels = reshaped.Shape[1];
        var sliceLength = reshaped.Shape.Stride(1); // Stride of each channel

        Tensor<float> result = Tensor<float>.Defaults(new TensorShape(batches, channels));
        for (int batch = 0, globalChannel = 0; batch < batches; batch++)
        {
            for (var channel = 0; channel < channels; channel++, globalChannel++)
            {
                float accumulator = default(float);
                int count = 0;

                var span = result.AsSpan(globalChannel * sliceLength, sliceLength);
                for (var i = 0; i < span.Length; i++)
                {
                    accumulator = Accumulate(accumulator, span[i], ++count);
                }

                result[batch, channel] = Aggregate(accumulator, count);
            }
        }

        return result;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var ishape = x.Shape.EnsureRank(4); // [Batch, Channel, Height, Width]
        var batches = ishape[0];
        var channels = ishape[1];
        var sliceLength = ishape.Stride(1);
        var height = ishape[2];
        var width = ishape[3];

        var dx = Tensor<float>.Zeros(ishape);

        for (int batch = 0, globalChannel = 0; batch < batches; batch++)
        {
            for (int channel = 0; channel < channels; channel++, globalChannel++)
            {
                // dy[batch, channel] is the gradient for the pooled output
                SpreadGradient(
                    x: x.AsSpan(globalChannel * sliceLength, sliceLength),
                    dx: dx.AsSpan(globalChannel * sliceLength, sliceLength),
                    dy: dy[batch, channel],
                    batch: batch,
                    channel: channel
                );
            }
        }

        return new Gradient(dx.ReshapeShared(x.Shape));
    }

    protected abstract void SpreadGradient(ReadOnlySpan<float> x, Span<float> dx, float dy, int batch, int channel);

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null) { }

}