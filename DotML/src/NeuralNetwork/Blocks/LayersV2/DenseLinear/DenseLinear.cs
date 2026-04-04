using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Fully connected layer which flattens it's inputs before processing and returns a column vector from the output neurons
/// <see href="https://en.wikipedia.org/wiki/Layer_(deep_learning)"/>
/// </summary>
public class DenseLinear : NetworkLayer, IWeightsAndBiasNetworkModule
{
    public int InputSize { get; init; }

    public int OutputSize => Neurons;

    public int Neurons { get; init; }

    private Tensor<float> _weights;
    public Tensor<float> Weights
    {
        get => _weights;
        set {
            // Setting safety check
            if (value.Shape != new Shape(Neurons, InputSize))
                throw new ArgumentException($"Weights must have shape ({Neurons}, {InputSize})");
            _weights = value;
        }
    }

    private Tensor<float> _biases;
    public Tensor<float> Biases
    {
        get => _biases;
        set {
            // Setting safety check
            if (value.Shape != new Shape(Neurons, 1) && value.Shape != new Shape(Neurons))
                throw new ArgumentException($"Biases must have shape ({Neurons}, 1), {value.Shape} given");
            _biases = value.ReshapeShared(new Shape(Neurons, 1)); // Ensure biases are always stored as column vector
        }
    }

    public DenseLinear(int input_size, int neurons)
    {
        this.InputSize = input_size;
        this.Neurons = neurons;

        _weights = Tensor<float>.Zeros(new Shape(neurons, input_size));
        _biases = Tensor<float>.Zeros(new Shape(neurons, 1));
    }

    public override int TrainableParameterCount()
    {
        return Weights.ElementCount + Biases.ElementCount;
    }

    public override void Initialize(IInitializer initializer)
    {
        var parameters = this.TrainableParameterCount();

        Weights.FillGenerated(() => initializer.RandomWeight(InputSize, OutputSize, parameters));
        Biases.FillGenerated(() => initializer.RandomBias(InputSize, OutputSize, parameters));
    }

    public override Shape ForwardShape(Shape input)
    {
        // See Forward
        var x2 = Flatten.FlattenNonBatch(input);

        // MatMul(Weights * X2) dimensions with batch dimensions kept as prefix
        int[] dims = new int[x2.Rank];
        dims[dims.Length - 1] = Weights.Shape.Length(^2);                        // b_cols Its a column vector as a result of flattening
        for (var i = 0; i < x2.Rank - 1; i++)
            dims[i] = x2.Length(i);

        return new Shape(dims);
    }

    public override Tensor<float> Forward(Tensor<float> x)
    {
        // Flatten the input to column/height dimension :[N, C, H, W] -> [N, F]
        x = Flatten.FlattenNonBatch(x);
        // Matrix multiplication, do broadcasting for batch dimensions as needed: [O, F] x [..., F]
        // Use the row dimension (^1) as the vector dimension
        var mul = Weights.MatMulEachVector(dimension: ^1, x, Biases.AsArray());

        // Add bias (broadcasting for batch dimensions as needed): [..., O] + [O]
        // Already handled by the above method

        return mul; // Output is [..., O] (IE it has a batch dimension)
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        // Flatten the input to column/height dimension: [...N, C, H, W] -> [N, 1, F, 1]
        var originalShape = x.Shape;
        x = x.ReshapeShared(x.Shape.EnsureRank(2));      // If input is [F] transform to [1, F]
        x = Flatten.FlattenNonBatch(x);                  // Flatten the training CHW dims [N, F]

        // Reshape x and dy for matmuls: [N, F], [N, O]
        var xFlatT = x;     // [N, F]
        var dyFlat = dy;    // [N, O]

        // Compute dW = sum_over_batch( dy * x^T ) => [O, F]
        var dW = dyFlat.TransposedMatMul(xFlatT); // [O, N] * [N, F] = [O, F]

        // Compute dB = sum_over_batch(dy) => [O, 1]
        var dB = dy.ReshapeShared(new Shape(dy.Shape.Length(0), OutputSize, 1))
            .Sum(axis: 0, keepdim: false);                        // [O, 1]

        // Compute dx = dy * W^T => [N, F]
        // [N, O] * [O, F] = [N, F]
        var dx = dyFlat.MatMul(Weights);

        // Reshape dx back to original input shape from the flattened format
        dx = dx.ReshapeShared(originalShape);

        return new WeightAndBiasGradients(dx, dW, dB);
    }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    {
        if (gradients is not WeightAndBiasGradients wbg)
            throw new ArgumentException("Expected WeightAndBiasGradients", nameof(gradients));

        var dW = wbg.dW;
        var dB = wbg.dB;

        // Apply regularization to weights
        if (regularization is not null)
        {
            dW.ElementWiseBinaryInplace(Weights, (gradient, prevWeight) => gradient + regularization.Invoke(prevWeight));
            dB.ElementWiseBinaryInplace(Biases, (gradient, prevWeight) => gradient + regularization.Invoke(prevWeight));
        }

        // Apply optimizer
        optimizer.UpdateParameter(this, nameof(Weights), learningRate, this.Weights, dW);
        optimizer.UpdateParameter(this, nameof(Biases), learningRate, this.Biases, dB);
    }

    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}