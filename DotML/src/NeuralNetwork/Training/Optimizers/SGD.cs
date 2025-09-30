namespace DotML.Network.Training;

public class SgdOptimizer : IOptimizer
{
    public void ClearCaches() { /* no caches */ }
    public Tensor<float> GetParameterUpdate(INetworkModule module, string name, float learningRate, Tensor<float> parameter, Tensor<float> gradient)
    {
        // Just subtract the gradient as is (base-class UpdateParameter handles the subtraction already)
        return gradient;
    }
    public void UpdateParameter(INetworkModule module, string name, float learningRate, Tensor<float> parameter, Tensor<float> gradient)
    {
        var update = GetParameterUpdate(module, name, learningRate, parameter, gradient);
        parameter.SubtractWithInplace(update);
    }
}