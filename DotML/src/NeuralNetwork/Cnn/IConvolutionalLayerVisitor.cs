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
public interface IConvolutionalLayerVisitor<TOut> {
    public TOut Visit(ConvolutionLayer layer);
    public TOut Visit(DepthwiseConvolutionLayer layer);
    public TOut Visit(PoolingLayer layer);
    public TOut Visit(FlatteningLayer layer);
    public TOut Visit(DropoutLayer layer);
    public TOut Visit(LayerNorm layer);
    public TOut Visit(FullyConnectedLayer layer);
    public TOut Visit(ActivationLayer layer);
    public TOut Visit(SoftmaxLayer layer);
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