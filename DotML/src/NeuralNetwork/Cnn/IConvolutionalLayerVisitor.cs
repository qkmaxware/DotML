namespace DotML.Network;

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer
/// </summary>
public interface IConvolutionalLayerVisitor {
    public void Visit(ConvolutionLayer layer);
    public void Visit(DepthwiseConvolutionLayer layer);
    public void Visit(PoolingLayer layer);
    public void Visit(FlatteningLayer layer);
    public void Visit(DropoutLayer layer);
    public void Visit(LayerNorm layer);
    public void Visit(FullyConnectedLayer layer);
    public void Visit(ActivationLayer layer);
    public void Visit(SoftmaxLayer layer);
}

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer
/// </summary>
public interface IConvolutionalLayerVisitor<T> {
    public T Visit(ConvolutionLayer layer);
    public T Visit(DepthwiseConvolutionLayer layer);
    public T Visit(PoolingLayer layer);
    public T Visit(FlatteningLayer layer);
    public T Visit(DropoutLayer layer);
    public T Visit(LayerNorm layer);
    public T Visit(FullyConnectedLayer layer);
    public T Visit(ActivationLayer layer);
    public T Visit(SoftmaxLayer layer);
}

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer with additional arguments
/// </summary>
public interface IConvolutionalLayerVisitor<TIn, TOut> {
    public TOut Visit(ConvolutionLayer layer, TIn args);
    public TOut Visit(DepthwiseConvolutionLayer layer, TIn args);
    public TOut Visit(PoolingLayer layer, TIn args);
    public TOut Visit(FlatteningLayer layer, TIn args);
    public TOut Visit(DropoutLayer layer, TIn args);
    public TOut Visit(LayerNorm layer, TIn args);
    public TOut Visit(FullyConnectedLayer layer, TIn args);
    public TOut Visit(ActivationLayer layer, TIn args);
    public TOut Visit(SoftmaxLayer layer, TIn args);
}