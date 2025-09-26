namespace DotML.Network.Training;

public class SgdOptimizer: IOptimizer {
    public void ClearCaches() { /* no caches */ }
    public Tensor<float> GetParameterUpdate(INetworkModule module, string name, Tensor<float> parameter, Tensor<float> gradient) {
        // Just subtract the gradient
        return gradient;
    }
}