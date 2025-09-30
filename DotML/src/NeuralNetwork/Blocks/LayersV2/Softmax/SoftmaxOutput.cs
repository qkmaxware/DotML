using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Softmax output layer for a FeedforwardNetwork
/// <see href="https://en.wikipedia.org/wiki/Softmax_function"/>
/// </summary>
public class SoftmaxOutput : NetworkLayer
{
    
    public Index ClassAxis { get; init; }

    public SoftmaxOutput() : this(^2) { }
    public SoftmaxOutput(Index classesAxis)
    {
        this.ClassAxis = classesAxis;
    }

    public override void Initialize(IInitializer initializer) { }

    public override TensorShape ForwardShape(TensorShape input) => input.EnsureRank(4);

    public override Tensor<float> Forward(Tensor<float> channels)
    {
        var originalRank = channels.Rank;
        var inShape = channels.Shape.EnsureRank(4); // Expecting NCHW (note this can be larger than 4, additional dimensions are just more batch dimensions)
        var inRank = inShape.Rank;
        int axis = this.ClassAxis.GetOffset(inRank);
        if (axis < 0 || axis >= inRank)
            throw new ArgumentOutOfRangeException(nameof(ClassAxis), $"Axis {axis} is out of bounds for shape with rank {inRank}.");

        if (this.IsTraining)
        {
            // Don't do any processing. ASSUME SOFTMAX IS DONE BY CROSS_ENTROPY LOSS
            return channels; // IDK what to do here anymore
        }

        int classCount = inShape.Length(axis);
        int batchSize = 1;
        for (int i = 0; i < inRank; i++)
        {
            if (i == axis) continue;
            batchSize *= inShape.Length(i);
        }
        int axisStride = inShape.Stride(axis);
        int batchStride = axisStride * classCount;

        var input = channels.AsSpan();
        var output = new float[input.Length];

        for (var batchIndex = 0; batchIndex < batchSize; batchIndex++)
        {
            int batchOffset = batchIndex * batchStride;

            var inputSpan = input.Slice(batchOffset, batchStride);
            var outputSpan = output.AsSpan(batchOffset, batchStride);

            // Compute max for numerical stability
            float max = inputSpan[0];
            for (int i = 1; i < classCount; i++)
            {
                float val = inputSpan[i * axisStride];
                if (val > max) max = val;
            }

            // Compute exponentials and sum
            float sum = 0f;
            for (int i = 0; i < classCount; i++)
            {
                float exp = MathF.Exp(inputSpan[i * axisStride] - max);
                outputSpan[i * axisStride] = exp;
                sum += exp;
            }

            // Normalize
            float invSum = 1.0f / sum;
            for (int i = 0; i < classCount; i++)
            {
                outputSpan[i * axisStride] *= invSum;
            }
        }

        return Tensor<float>.FromFlattenedArray(inShape, output).Squeeze(0..^originalRank);
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy) {
        // Do nothing, just pass back the error. ASSUMING THIS IS HANDLED BY CROSS_ENTOPY LOSS
        return new Gradient(dy);
    }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)  { /* Nothing to do here */ }
}