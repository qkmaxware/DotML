using System.Collections;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Base interface for all CNN layers
/// </summary>
public interface IFeedforwardNetworkLayer : ILayer {
    public FeatureSet<double> EvaluateSync(FeatureSet<double> features);
    public BatchedFeatureSet<double> EvaluateSync(BatchedFeatureSet<double> features);
    public BackpropagationReturns Backpropagate(BackpropagationArgs args);

    public void BeginTraining();
    public void EndTraining();

    public void Initialize(IInitializer initializer);

    public bool DoesShapeMatchInputShape(Shape3D shape);

    public void Visit(ILayerVisitor visitor);
    public T Visit<T>(ILayerVisitor<T> visitor);
    public TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args);
}

/// <summary>
/// Base class for all layers for a CNN
/// </summary>
public abstract class FeedforwardNetworkLayer : IFeedforwardNetworkLayer {
    public virtual Shape3D InputShape {get; protected set;}
    public virtual Shape3D OutputShape {get; protected set;}

    public virtual bool DoesShapeMatchInputShape(Shape3D shape) {
        return shape.Channels == InputShape.Channels 
            && InputShape.Rows == shape.Rows
            && InputShape.Columns == shape.Columns
        ;
    }

    public bool IsTraining {get; private set;}
    public bool IsInference => !IsTraining;

    public void BeginTraining() {
        IsTraining = true;
        OnTrainingBegin();
    }
    public void EndTraining() {
        IsTraining = false;
        OnTrainingEnd();
    }
    protected virtual void OnTrainingBegin() {}
    protected virtual void OnTrainingEnd() {}

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

    public abstract BackpropagationReturns Backpropagate(BackpropagationArgs args);

    public abstract void Visit(ILayerVisitor visitor);
    public abstract T Visit<T>(ILayerVisitor<T> visitor);
    public abstract TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args);

    public LayerSequencer Then(NetworkLayerGenerator constructor) {
        var seq = new LayerSequencer(this);
        return seq.Then(constructor);
    }
}