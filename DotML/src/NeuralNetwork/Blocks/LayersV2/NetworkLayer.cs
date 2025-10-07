using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Base class for all neural network layers
/// <see href="https://en.wikipedia.org/wiki/Layer_(deep_learning)"/>
/// </summary>
public abstract class NetworkLayer : INetworkModule, IBlockVisitable
{
    /// <summary>
    /// Pre-configured parallel executions options for Parallel.For or other uses in child classes
    /// </summary>
    protected static ParallelOptions ParallelOptions = new ParallelOptions
    {
        // TODO if we need better determination of max threads, put the logic here
        MaxDegreeOfParallelism = Tensor<float>.DegreesOfParallelism
    };

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
    /// Feed-forward a given tensor shape and return the expected output shape
    /// </summary>
    /// <param name="input">input tensor shape</param>
    /// <returns>output tensor shape</returns>
    public abstract TensorShape ForwardShape(TensorShape input);

    /// <summary>
    /// Feed-forward a given tensor to this layer
    /// </summary>
    /// <param name="channels">layer input</param>
    /// <returns>layer output</returns>
    public abstract Tensor<float> Forward(Tensor<float> channels);

    public virtual Tensor<float> Forward(Tensor<float> channels, EvaluationContext? ctx)
    {
        var res = this.Forward(channels);
        if (ctx is not null)
        {
            ctx.Save(this, new IOContext(channels, res));
        }
        return res;
    }

    /// <summary>
    /// Backwards evaluation of this layer
    /// </summary>
    /// <param name="x">original input</param>
    /// <param name="y">original output</param>
    /// <param name="dy">gradient of loss w.r.t output</param>
    /// <returns>gradient of loss w.r.t input and layer specific gradients if applicable</returns>
    public abstract Gradients Backward(Tensor<float> x, Tensor<float> y, Tensor<float> dy);

    public virtual Gradients Backward(Tensor<float> dy, EvaluationContext ctx, IClippingStrategy? clipping = null)
    {
        var io = ctx.Get<IOContext>(this);
        var grads = this.Backward(io.Input, io.Output, dy);
        if (clipping is not null)
            grads.Clip(clipping);
        return grads;
    }

    public abstract void Update(float learningRate, Gradients gradients, IOptimizer optimizer, RegularizationFunction? regularization = null);

    /// <summary>
    /// Visit this layer with a given visitor
    /// </summary>
    /// <param name="visitor">visitor</param>
    /// <param name="arg">optional argument</param>
    /// <returns>optional return</returns>
    public abstract TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg);
}