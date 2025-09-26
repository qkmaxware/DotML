using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

public interface INetworkModule {
    public void Initialize(IInitializer initializer);
    public Tensor<float> Forward(Tensor<float> channels, EvaluationContext? ctx = null);
    public Gradients Backward(Tensor<float> dy, EvaluationContext ctx, IClippingStrategy? clipping = null);
    public void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null);
}