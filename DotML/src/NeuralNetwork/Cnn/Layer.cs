using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Base interface for all CNN layers
/// </summary>
public interface IConvolutionalFeedforwardNetworkLayer : ILayer {
    public Matrix<double>[] EvaluateSync(Matrix<double>[] channels);

    public void Initialize(IInitializer initializer);

    public void Visit(IConvolutionalLayerVisitor visitor);
    public T Visit<T>(IConvolutionalLayerVisitor<T> visitor);
    public TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args);
}

/// <summary>
/// Base class for all layers for a CNN
/// </summary>
public abstract class ConvolutionalFeedforwardNetworkLayer : IConvolutionalFeedforwardNetworkLayer {
    public abstract int InputCount {get;}
    public abstract int OutputCount {get;}
    public abstract int NeuronCount {get;}

    public Vec<double> GetLastOutputs() {
        throw new NotImplementedException();
    }

    public abstract void Initialize(IInitializer initializer);

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public abstract int TrainableParameterCount();

    /// <summary>
    /// Evaluate the output of the layer when applied to the given input image
    /// </summary>
    /// <param name="channels">Input image represented in channels </param>
    /// <returns>output channel values</returns>
    public abstract Matrix<double>[] EvaluateSync(Matrix<double>[] channels);

    public abstract void Visit(IConvolutionalLayerVisitor visitor);
    public abstract T Visit<T>(IConvolutionalLayerVisitor<T> visitor);
    public abstract TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args);
}