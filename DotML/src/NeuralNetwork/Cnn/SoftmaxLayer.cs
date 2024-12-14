using System.Drawing;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Softmax output layer for a ConvolutionalFeedforwardNetwork
/// </summary>
public class SoftmaxLayer : ConvolutionalFeedforwardNetworkLayer {

    private int size;

    public SoftmaxLayer(int size) {
        this.size = size;

        this.InputShape = new Shape3D(1, size, 1);
        this.OutputShape = new Shape3D(1, size, 1);
    }

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] inputs) {
        // Treat all inputs values as a single vector, compute the softmax of this vector
        var sum = 0.0d;
        double[,] values = new double[size, 1];
        var i = 0;
        foreach (var input in inputs) {
            foreach (var item in input) {
                var exp_i = Math.Exp(item);
                values[i++, 0] = exp_i;
                sum += exp_i;
            }
        }
        for (var j = 0; j < size; j++) {
            values[j, 0] = values[j, 0] / sum;
        }

        return [ Matrix<double>.Wrap(values) ];
    }

    public override void Initialize(IInitializer initializer) { /* No initialization needed */ }

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public override int TrainableParameterCount() => 0;

    public override void Visit(IConvolutionalLayerVisitor visitor) {
        visitor.Visit(this);
    }

    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) {
        return visitor.Visit(this);
    }

    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) {
        return visitor.Visit(this, args);
    }
}