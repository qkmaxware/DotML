using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;


public abstract class Reshape : NetworkLayer
{
    public override void Initialize(IInitializer initializer) { }

    public override void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null)  { /* Nothing to do here */ }

    public override Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy)
    {
        return new Gradient(dy.ReshapeShared(x.Shape));
    }
}