using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Base class for all neural network layers
/// <see href="https://en.wikipedia.org/wiki/Layer_(deep_learning)"/>
/// </summary>
public abstract class NetworkLayer
{
    /// <summary>
    /// Test if the layer is in training mode (defaults to false, inference mode)
    /// </summary>
    public bool IsTraining { get; private set; } = false;

    /// <summary>
    /// Test if the layer is in inference mode
    /// </summary>
    public bool IsInference => !IsTraining;

    /// <summary>
    /// Configure this layer for training
    /// </summary>
    public void BeginTraining()
    {
        IsTraining = true;
        OnTrainingBegin();
    }

    /// <summary>
    /// Configure this layer for inference
    /// </summary>
    public void EndTraining()
    {
        IsTraining = false;
        OnTrainingEnd();
    }
    protected virtual void OnTrainingBegin() { }
    protected virtual void OnTrainingEnd() { }

    /// <summary>
    /// Initialize the layer with the given initialization strategy
    /// </summary>
    /// <param name="initializer">initialization strategy</param>
    public abstract void Initialize(IInitializer initializer);

    /// <summary>
    /// Number of un-trainable parameters
    /// </summary>
    /// <returns>count</returns>
    public virtual int UnTrainableParameterCount() => 0;

    /// <summary>
    /// Number of trainable parameters
    /// </summary>
    /// <returns>count</returns>
    public virtual int TrainableParameterCount() => 0;

    /// <summary>
    /// Feed-forward a given tensor to this layer
    /// </summary>
    /// <param name="channels">layer input</param>
    /// <returns>layer output</returns>
    public abstract Tensor<float> Forward(Tensor<float> channels);

    /// <summary>
    /// Backwards evaluation of this layer
    /// </summary>
    /// <param name="x">original input</param>
    /// <param name="y">original output</param>
    /// <param name="dy">gradient of loss w.r.t output</param>
    /// <returns>gradient of loss w.r.t input and layer specific gradients if applicable</returns>
    public abstract Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy);

    /// <summary>
    /// Update this layer's weights and biases used the provided gradients usually computed via a call to <see cref="Backward"/>
    /// </summary>
    /// <param name="grads">layer specific gradients</param>
    public abstract void SubtractGradients(Gradients grads);
}