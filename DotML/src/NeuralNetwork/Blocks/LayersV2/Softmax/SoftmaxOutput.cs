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

    public override TensorShape ForwardShape(TensorShape input) => input;

    public override Tensor<float> Forward(Tensor<float> channels) => channels.Softmax(this.ClassAxis); // Softmax during inference
    public override Tensor<float> Forward(Tensor<float> channels, EvaluationContext? ctx)
    {
        if (ctx is not null && ctx.Mode == EvaluationMode.Training)
            return channels; // Do nothing during training. ASSUME SOFTMAX IS DONE BY CROSS_ENTROPY LOSS

        return channels.Softmax(this.ClassAxis); // Softmax during inference
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        // Do nothing, just pass back the error. ASSUMING THIS IS HANDLED BY CROSS_ENTOPY LOSS
        return new Gradient(dy);
    }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null) { /* Nothing to do here */ }

    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg) => visitor.Visit(this, arg);
}