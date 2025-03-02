namespace DotML.Network;

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer
/// </summary>
public interface ILayerVisitor {
    public void Visit(ConvolutionLayer layer);
    public void Visit(DepthwiseConvolutionLayer layer);
    public void Visit(PoolingLayer layer);
    public void Visit(FlatteningLayer layer);
    public void Visit(DropoutLayer layer);
    public void Visit(LayerNorm layer);
    public void Visit(BatchNorm layer);
    public void Visit(DenseLinearLayer layer);
    public void Visit(ActivationLayer layer);
    public void Visit(SoftmaxLayer layer);

    public void Visit(InputCapture capture);
}

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer
/// </summary>
public interface ILayerVisitor<TOut> {
    public TOut Visit(ConvolutionLayer layer);
    public TOut Visit(DepthwiseConvolutionLayer layer);
    public TOut Visit(PoolingLayer layer);
    public TOut Visit(FlatteningLayer layer);
    public TOut Visit(DropoutLayer layer);
    public TOut Visit(LayerNorm layer);
    public TOut Visit(BatchNorm layer);
    public TOut Visit(DenseLinearLayer layer);
    public TOut Visit(ActivationLayer layer);
    public TOut Visit(SoftmaxLayer layer);

    public TOut Visit(InputCapture capture);
}

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer with additional arguments
/// </summary>
public interface ILayerVisitor<TIn, TOut> {
    public TOut Visit(ConvolutionLayer layer, TIn args);
    public TOut Visit(DepthwiseConvolutionLayer layer, TIn args);
    public TOut Visit(PoolingLayer layer, TIn args);
    public TOut Visit(FlatteningLayer layer, TIn args);
    public TOut Visit(DropoutLayer layer, TIn args);
    public TOut Visit(LayerNorm layer, TIn args);
    public TOut Visit(BatchNorm layer, TIn args);
    public TOut Visit(DenseLinearLayer layer, TIn args);
    public TOut Visit(ActivationLayer layer, TIn args);
    public TOut Visit(SoftmaxLayer layer, TIn args);

    public TOut Visit(InputCapture capture, TIn args);
}