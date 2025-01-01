using System.Collections;
using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Base interface for all CNN layers
/// </summary>
public interface IConvolutionalFeedforwardNetworkLayer : ILayer {
    public FeatureSet<double> EvaluateSync(FeatureSet<double> features);
    public BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features);

    public void Initialize(IInitializer initializer);

    public bool DoesShapeMatchInputShape(Matrix<double>[] channels);

    public void Visit(IConvolutionalLayerVisitor visitor);
    public T Visit<T>(IConvolutionalLayerVisitor<T> visitor);
    public TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args);
}

/// <summary>
/// Base class for all layers for a CNN
/// </summary>
public abstract class ConvolutionalFeedforwardNetworkLayer : IConvolutionalFeedforwardNetworkLayer {
    public virtual Shape3D InputShape {get; protected set;}
    public virtual Shape3D OutputShape {get; protected set;}

    public virtual bool DoesShapeMatchInputShape(Matrix<double>[] channels) {
        return channels.Length == InputShape.Channels 
            && InputShape.Rows == channels.FirstOrDefault().Rows
            && InputShape.Columns == channels.FirstOrDefault().Columns
        ;
    }

    public abstract void Initialize(IInitializer initializer);

    /// <summary>
    /// Number of trainable parameters in this layer
    /// </summary>
    /// <returns>Number of trainable parameters</returns>
    public abstract int TrainableParameterCount();

    /// <summary>
    /// Number of un-trainable parameters in this layer
    /// </summary>
    /// <returns>Number of un-trainable parameters</returns>
    public virtual int UnTrainableParameterCount() => 0;

    /// <summary>
    /// Evaluate the output of the layer when applied to the given input image
    /// </summary>
    /// <param name="channels">Input image represented in channels </param>
    /// <returns>output channel values</returns>
    public abstract FeatureSet<double> EvaluateSync(FeatureSet<double> channels);

    public virtual BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features) {
        var results = new FeatureSet<double>[features.Batches];
        Parallel.For(0, results.Length, i => {
            results[i] = EvaluateSync(features[i]);
        });
        return new BatchedFeatureSet<double>(results);
    }

    public abstract void Visit(IConvolutionalLayerVisitor visitor);
    public abstract T Visit<T>(IConvolutionalLayerVisitor<T> visitor);
    public abstract TOut Visit<TIn, TOut>(IConvolutionalLayerVisitor<TIn, TOut> visitor, TIn args);

    public LayerSequencer Then(Func<Shape3D, IConvolutionalFeedforwardNetworkLayer> constructor) {
        var seq = new LayerSequencer(this);
        return seq.Then(constructor);
    }
}

public class LayerSequencer : IEnumerable<IConvolutionalFeedforwardNetworkLayer> {
    private Shape3D ishape;
    private List<IConvolutionalFeedforwardNetworkLayer> layers = new List<IConvolutionalFeedforwardNetworkLayer>();

    public Shape3D InputShape => layers.Count > 0 ? layers[0].InputShape : ishape;
    public Shape3D OutputShape => layers.Count > 0 ? layers[^1].OutputShape : ishape;

    public LayerSequencer(Shape3D input_size) {
        this.ishape = input_size;
    } 

    public LayerSequencer(IConvolutionalFeedforwardNetworkLayer first) {
        this.layers.Add(first);
    }

    public LayerSequencer Then(Func<Shape3D, IConvolutionalFeedforwardNetworkLayer> constructor) {
        this.layers.Add(constructor(OutputShape));
        return this;
    }

    public LayerSequencer Then(Func<Shape3D, IEnumerable<IConvolutionalFeedforwardNetworkLayer>> generator) {
        this.layers.AddRange(generator(OutputShape));
        return this;
    }

    public IEnumerator<IConvolutionalFeedforwardNetworkLayer> GetEnumerator() => layers.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => layers.GetEnumerator();
}