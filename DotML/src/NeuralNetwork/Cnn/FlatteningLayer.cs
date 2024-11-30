using System.Runtime.InteropServices;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Layer which flattens inputs into a column vector 
/// </summary>
public class FlatteningLayer : ConvolutionalFeedforwardNetworkLayer {
    public override int InputCount => -1;
    public override int OutputCount => -1;
    public override int NeuronCount => -1;

    public override void Initialize(IInitializer initializer) { }
    public override int TrainableParameterCount() => 0;

    public override Matrix<double>[] EvaluateSync(Matrix<double>[] inputs) {
        // input is a 2D matrix processed from prior layers like a pooling layer
        var x = inputs.Length == 1 && inputs[0].IsColumn ? inputs[0] : Matrix<double>.Column(inputs.SelectMany(x => x.FlattenRows()).ToArray());
        return [ x ];
    }

     public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}