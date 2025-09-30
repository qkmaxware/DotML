using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Activation layer for a FeedforwardNetwork
/// <see href="https://en.wikipedia.org/wiki/Activation_function"/>
/// </summary>
public class ActivationLayer2 : NetworkLayer
{
    public ActivationFunction ActivationFunction { get; init; }

    public ActivationLayer2(ActivationFunction activation)
    {
        this.ActivationFunction = activation;
    }

    public override void Initialize(IInitializer initializer) { }

    public override int UnTrainableParameterCount() => 0;

    public override int TrainableParameterCount() => 0;

    public override TensorShape ForwardShape(TensorShape input) => input;

    public override Tensor<float> Forward(Tensor<float> channels)
    {
        return channels.ElementWise(ActivationFunction.Invoke);
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        var dx = x.ElementWise(ActivationFunction.InvokeDerivative);
        dx.HadamardWithInplace(dy);
        return new Gradient(dx);
    }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)  { /* Nothing to do here */ }
}