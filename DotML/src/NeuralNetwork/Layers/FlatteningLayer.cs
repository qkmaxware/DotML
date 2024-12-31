using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Layer which flattens inputs into a column vector 
/// </summary>
[Untested()]
public class FlatteningLayer : ConvolutionalFeedforwardNetworkLayer {
    public override void Initialize(IInitializer initializer) { }
    public override int TrainableParameterCount() => 0;

    public FlatteningLayer(Shape3D input_size) {
        this.InputShape = input_size;
        this.OutputShape = new Shape3D(1, input_size.Count, 1);
    }

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> inputs) {
        // input is a 2D matrix processed from prior layers like a pooling layer
        var x = inputs.Channels == 1 && inputs[0].IsColumn ? inputs[0] : Matrix<double>.Column(inputs.SelectMany(x => x.FlattenRows()).ToArray());
        return new FeatureSet<double>(x);
    }

     public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);
}