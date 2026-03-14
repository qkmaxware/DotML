namespace DotML.Network.Training;

/// <summary>
/// A parameter optimizer
/// </summary>
public interface IOptimizer
{
    /// <summary>
    /// Clear any internal caches the optimizer may use
    /// </summary>
    public void ClearCaches();

    /// <summary>
    /// Compute the parameter update that should be applied to the parameter given the gradient
    /// </summary>
    /// <param name="module">module the parameter applies to</param>
    /// <param name="name">the name of the parameter</param>
    /// <param name="parameter">the parameter</param>
    /// <param name="gradient">the parameter's gradients</param>
    /// <returns>parameter update tensor</returns>
    public Tensor<float> GetParameterUpdate(INetworkModule module, string name, float learningRate, Tensor<float> parameter, Tensor<float> gradient);

    /// <summary>
    /// Computes and applies the parameter update given the gradient of the parameter. Similar to <see cref="GetParameterUpdate"/> but also applies it to the parameter by modifying the parameter tensor in-place with the computed update.
    /// </summary>
    /// <param name="module">module the parameter applies to</param>
    /// <param name="name">the name of the parameter</param>
    /// <param name="parameter">the parameter</param>
    /// <param name="gradient">the parameter's gradients</param>
    public void UpdateParameter(INetworkModule module, string name, float learningRate, Tensor<float> parameter, Tensor<float> gradient)
    {
        var update = GetParameterUpdate(module, name, learningRate, parameter, gradient);
        parameter.SubtractWithInplace(update);
    }
}