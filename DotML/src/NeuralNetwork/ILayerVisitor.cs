namespace DotML.Network;

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer
/// </summary>
public interface ILayerVisitor {
    public void Visit(ConvolutionLayer layer);
    public void Visit(DepthwiseConvolutionLayer layer);
    public void Visit(TransposeConvolutionLayer layer);
    public void Visit(PixelShuffle layer);
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
public interface ILayerOutputVisitor<TOut> {
    public TOut Visit(ConvolutionLayer layer);
    public TOut Visit(DepthwiseConvolutionLayer layer);
    public TOut Visit(TransposeConvolutionLayer layer);
    public TOut Visit(PixelShuffle layer);
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
public interface ILayerInputVisitor<TIn> {
    public void Visit(ConvolutionLayer layer, TIn args);
    public void Visit(DepthwiseConvolutionLayer layer, TIn args);
    public void Visit(TransposeConvolutionLayer layer, TIn args);
    public void Visit(PixelShuffle layer, TIn args);
    public void Visit(PoolingLayer layer, TIn args);
    public void Visit(FlatteningLayer layer, TIn args);
    public void Visit(DropoutLayer layer, TIn args);
    public void Visit(LayerNorm layer, TIn args);
    public void Visit(BatchNorm layer, TIn args);
    public void Visit(DenseLinearLayer layer, TIn args);
    public void Visit(ActivationLayer layer, TIn args);
    public void Visit(SoftmaxLayer layer, TIn args);
    public void Visit(InputCapture capture, TIn args);
}

/// <summary>
/// An object that can apply a different Visit method for each type of convolutional layer with additional arguments
/// </summary>
public interface ILayerInputOutputVisitor<TIn, TOut> {
    public TOut Visit(ConvolutionLayer layer, TIn args);
    public TOut Visit(DepthwiseConvolutionLayer layer, TIn args);
    public TOut Visit(TransposeConvolutionLayer layer, TIn args);
    public TOut Visit(PixelShuffle layer, TIn args);
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