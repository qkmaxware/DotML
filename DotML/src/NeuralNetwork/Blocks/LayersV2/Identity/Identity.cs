using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// The identity layer, otherwise known as the identity function, does nothing to its inputs. 
/// </summary>
public class Identity : NetworkLayer
{
    public override TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg)
    {
        return visitor.Visit(this, arg);
    }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        // Just pass back the gradients
        return new Gradient(dy);
    }

    public override Tensor<float> Forward(Tensor<float> channels)
    {
        // Just pass the inputs forward
        return channels;
    }

    public override Shape ForwardShape(Shape input)
    {
        // The output is always the same shape as the input because this layer DOES NOTHING!
        return input;
    }

    public override void Initialize(IInitializer initializer)
    { }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)
    { }
}