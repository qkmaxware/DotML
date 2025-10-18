using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Interface for all network modules (layer, network, subnetwork etc...)
/// </summary>
public interface INetworkModule
{
    /// <summary>
    /// Initialize the layer with the given initialization strategy
    /// </summary>
    /// <param name="initializer">initialization strategy</param>
    public void Initialize(IInitializer initializer);

    /// <summary>
    /// Feed-forward a given tensor shape and return the expected output shape
    /// </summary>
    /// <param name="input">input tensor shape</param>
    /// <returns>output tensor shape</returns>
    public TensorShape ForwardShape(TensorShape input);

    /// <summary>
    /// Feed-forward a given tensor to this layer
    /// </summary>
    /// <param name="channels">layer input</param>
    /// <param name="ctx">optional evaluation context for caching intermediary tensors</param>
    /// <returns>layer output</returns>
    public Tensor<float> Forward(Tensor<float> channels, EvaluationContext? ctx = null);

    /// <summary>
    /// Backwards evaluation of this layer
    /// </summary>
    /// <param name="dy">gradient of loss w.r.t output</param>
    /// <param name="ctx">evaluation context with cached intermediary tensors</param>
    /// <param name="clipping">optional gradient clipping strategy</param>
    /// <returns>gradient of loss w.r.t input and layer specific gradients if applicable</returns>
    public Gradients Backward(Tensor<float> dy, EvaluationContext ctx, IClippingStrategy? clipping = null);

    /// <summary>
    /// Update this layer's weights and biases used the provided gradients usually computed via a call to <see cref="Backward"/>
    /// </summary>
    /// <param name="learningRate">base learning rate</param>
    /// <param name="gradients">gradients for this module</param>
    /// <param name="optimizer">optimizer to apply the update to parameters</param>
    /// <param name="regularization">regularization strategy</param>
    public void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null);
}

/// <summary>
/// A network module that exposes it's weights and biases
/// </summary>
public interface IWeightsAndBiasNetworkModule
: INetworkModule
{
    public Tensor<float> Weights { get; set; }
    public Tensor<float> Biases { get; set; }
}