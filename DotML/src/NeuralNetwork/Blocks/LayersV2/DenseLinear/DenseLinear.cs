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

        Weights = Tensor<float>.Zeros(new TensorShape(neurons, input_size));
        Biases = Tensor<float>.Zeros(new TensorShape(neurons, 1));
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

    public override TensorShape ForwardShape(TensorShape input)
    {
        // See Forward
        var x2 = Flatten.FlattenCHW2H(input);
        // See Tensor.BatchedMatMul
        var broadcast_shape = TensorShape.ComputeBroadcastShape(this.Weights.Shape, x2);
        var a = Weights.Shape.BroadcastTo(broadcast_shape, 0, broadcast_shape.Rank - 2);
        var b = x2.BroadcastTo(broadcast_shape, 0, broadcast_shape.Rank - 2);

        var a_rank = a.Rank;
        var b_rank = b.Rank;

        // Last 2 dims must be matrix multiplication compatible
        var a_row_idx = a_rank - 2;
        var a_col_idx = a_rank - 1;
        var b_row_idx = b_rank - 2;
        var b_col_idx = b_rank - 1;
        int a_rows = a.Length(a_row_idx);
        int a_cols = a.Length(a_col_idx);
        int b_rows = b.Length(b_row_idx);
        int b_cols = b.Length(b_col_idx);

        if (a_cols != b_rows)
            throw new InvalidOperationException("Inner dimensions are not compatible for matrix multiplication");

        int r_matsize = a_rows * b_cols;

        int[] r_shape = new int[broadcast_shape.Rank]; // The actual shape of the output (most copied from the broadcast shape, the last 2 from mat-mul)
        r_shape[r_shape.Length - 2] = a_rows;
        r_shape[r_shape.Length - 1] = b_cols;
        int r_count = r_matsize;
        for (var i = 0; i < r_shape.Length - 2; i++)
        {
            var dim_length = broadcast_shape.Length(i);
            r_shape[i] = dim_length;
            r_count *= dim_length;
        }
        return new TensorShape(r_shape);
    }

    public override Tensor<float> Forward(Tensor<float> x)
    {
        // Flatten the input to column/height dimension :[N, C, H, W] -> [N, 1, F, 1]
        x = Flatten.FlattenCHW2H(x);

        // Matrix multiplication, do broadcasting for batch dimensions as needed: [O, F] x [..., F, 1]
        var mul = Weights.BatchedMatMul(x);

        // Add bias (broadcasting for batch dimensions as needed): [..., O, 1] + [O, 1]
        mul.AddWithInplace(Biases);

        return mul;
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        // Flatten the input to column/height dimension: [...N, C, H, W] -> [N, 1, F, 1]
        x = x.ReshapeShared(x.Shape.NormalizeRank(4));  // Collapse all leading batch dims into 1
        x = Flatten.FlattenCHW2H(x);                    // Flatten the training CHW dims

        // Reshape x and dy for matmuls: [N, F, 1], [N, O, 1]
        var xFlatT = x.ReshapeShared(new TensorShape(x.Shape.Length(0), 1, InputSize));     // [N, 1, F]
        var dyFlat = dy.ReshapeShared(new TensorShape(dy.Shape.Length(0), OutputSize, 1));  // [N, O, 1]

        // Compute dW = sum_over_batch( dy * x^T ) => [O, F]
        var dW = dyFlat.BatchedMatMul(xFlatT)                              // [N, O, F]
                    .Sum(axis: 0);                                         // [O, F]

        // Compute dB = sum_over_batch(dy) => [O, 1]
        var dB = dyFlat.Sum(axis: 0);                                      // [O, 1]

        // Compute dx = dy * W^T => [N, F, 1]
        var WT = Weights.Transpose();                                      // [F, O]
        var dx = dyFlat.BatchedMatMul(WT);                                 // [N, F, 1]

        // Reshape dx back to original input shape from the flattened format
        dx = dx.ReshapeShared(x.Shape);

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
}