namespace DotML.Network.Training;

public interface IOptimizer {
    public void ClearCaches();

    public Tensor<float> GetParameterUpdate(INetworkModule module, string name, Tensor<float> parameter, Tensor<float> gradient);
    public void UpdateParameter(INetworkModule module, string name, Tensor<float> parameter, Tensor<float> gradient) {
        var update = GetParameterUpdate(module, name, parameter, gradient);
        parameter.SubtractWithInplace(update);
    }
}