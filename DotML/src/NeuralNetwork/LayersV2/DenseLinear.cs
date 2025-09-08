using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Fully connected layer which flattens it's inputs before processing and returns a column vector from the output neurons
/// <see href="https://en.wikipedia.org/wiki/Layer_(deep_learning)"/>
/// </summary>
public class DenseLinear : NetworkLayer
{
    public int InputSize { get; init; }

    public int OutputSize => Neurons;

    public int Neurons { get; init; }

    public Tensor<float> Weights { get; private set; }

    public Tensor<float> Biases { get; private set; }

    public DenseLinear(int input_size, int neurons)
    {
        this.InputSize = input_size;
        this.Neurons = neurons;

        Weights = Tensor<float>.Defaults(new TensorShape(neurons, input_size));
        Biases = Tensor<float>.Defaults(new TensorShape(neurons, 1));
    }

    public override int TrainableParameterCount()
    {
        return Weights.ElementCount + Biases.ElementCount;
    }

    public override void Initialize(IInitializer initializer)
    {
        var parameters = this.TrainableParameterCount();

        Weights.FillGenerated(() => initializer.RandomWeight(InputSize, OutputSize, parameters));
        Biases.FillGenerated(() => initializer.RandomWeight(InputSize, OutputSize, parameters));
    }

    public override Tensor<float> Forward(Tensor<float> x)
    {
        // Flatten the input to column (NCHW) -> N1F1
        var xShape = x.Shape.EnsureRank(4);             // Rank < 4, pad with 1's
        var dims = new int[xShape.Rank];                // Copy batch dimensions
        dims[^3] = 1;
        dims[^2] = xShape.Length(^3) * xShape.Length(^2) * xShape.Length(^1);
        dims[^1] = 1;
        for (var i = 0; i < dims.Length - 3; i++)
            dims[i] = xShape.Length(i);              
        x = x.ReshapeShared(new TensorShape(dims));     // Flatten to column, reuse same data array

        // Matrix multiplication, do broadcasting for batch dimensions as needed
        var mul = Weights.BatchedMatMul(x);

        // Add bias (broadcasting for batch dimensions as needed)
        mul.AddWithInplace(Biases);

        return mul;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        throw new NotImplementedException();
    }

    public override void SubtractGradients(Gradients grads)
    {
        if (grads is not WeightAndBiasGradients wbg)
            throw new ArgumentException("Expected WeightAndBiasGradients", nameof(grads));

        this.Weights.SubtractWithInplace(wbg.dW);
        this.Biases.SubtractWithInplace(wbg.dB);
    }
}