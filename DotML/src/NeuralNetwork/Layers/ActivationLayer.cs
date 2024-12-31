using System.Drawing;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Activation layer for a ConvolutionalFeedforwardNetwork
/// <see href="https://en.wikipedia.org/wiki/Activation_function"/>
/// </summary>
public class ActivationLayer : ConvolutionalFeedforwardNetworkLayer {

    public ActivationFunction ActivationFunction {get; init;}

    public ActivationLayer(Shape3D input_size, ActivationFunction activation) {
        this.InputShape = input_size;
        this.OutputShape = input_size;
        this.ActivationFunction = activation;
    }

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() => 0;

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        var len = channels.Channels;
        Matrix<double>[] outputs = new Matrix<double>[len];
        Parallel.For(0, len, i => {
            outputs[i] = channels[i].Transform(ActivationFunction.Invoke);
        });
        return (FeatureSet<double>)outputs;
    }

    public override void Visit(IConvolutionalLayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(IConvolutionalLayerVisitor<T> visitor) =>visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

    public override string ToString() {
        return base.ToString() + "(" + ActivationFunction.ToString() + ")";
    }
}