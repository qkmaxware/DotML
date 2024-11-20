namespace DotML.Network;

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer
/// </summary>
public interface IConvolutionalLayerVisitor {
    public void Visit(ConvolutionLayer layer);
    public void Visit(PoolingLayer layer);
    public void Visit(FullyConnectedLayer layer);
    public void Visit(SoftmaxLayer layer);
}

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer
/// </summary>
public interface IConvolutionalLayerVisitor<T> {
    public T Visit(ConvolutionLayer layer);
    public T Visit(PoolingLayer layer);
    public T Visit(FullyConnectedLayer layer);
    public T Visit(SoftmaxLayer layer);
}

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer with additional arguments
/// </summary>
public interface IConvolutionalLayerVisitor<TIn, TOut> {
    public TOut Visit(ConvolutionLayer layer, TIn args);
    public TOut Visit(PoolingLayer layer, TIn args);
    public TOut Visit(FullyConnectedLayer layer, TIn args);
    public TOut Visit(SoftmaxLayer layer, TIn args);
}