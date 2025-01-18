using System.Drawing;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Softmax output layer for a FeedforwardNetwork
/// <see href="https://en.wikipedia.org/wiki/Softmax_function"/>
/// </summary>
public class SoftmaxLayer : FeedforwardNetworkLayer {

    public int Size {get; init;}

    public SoftmaxLayer(int size) {
        this.Size = size;

        this.InputShape = new Shape3D(1, size, 1);
        this.OutputShape = new Shape3D(1, size, 1);
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> inputs) {
        // Treat all inputs values as a single vector, compute the softmax of this vector
        var sum = 0.0d;
        double[,] values = new double[Size, 1];
        var i = 0;
        foreach (var input in inputs) {
            foreach (var item in input) {
                var exp_i = Math.Exp(item);
                values[i++, 0] = exp_i;
                sum += exp_i;
            }
        }
        for (var j = 0; j < Size; j++) {
            values[j, 0] = values[j, 0] / sum;
        }

        return new FeatureSet<double>( Matrix<double>.Wrap(values) );
    }

    public override void Initialize(IInitializer initializer) { /* No initialization needed */ }

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public override int TrainableParameterCount() => 0;

    public override void Visit(ILayerVisitor visitor) {
        visitor.Visit(this);
    }

    public override T Visit<T>(ILayerVisitor<T> visitor) {
        return visitor.Visit(this);
    }

    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) {
        return visitor.Visit(this, args);
    }
}