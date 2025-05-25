using System.Collections;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Base interface for all CNN layers
/// </summary>
public interface IFeedforwardNetworkLayer : ILayer {
    public FeatureSet<float> EvaluateSync(FeatureSet<float> features);
    public BatchedFeatureSet<float> EvaluateSync(BatchedFeatureSet<float> features);
    public BackpropagationReturns Backpropagate(BackpropagationArgs args);
    public void SubtractGradients(LayerGradients? gradients);

    public void BeginTraining();
    public void EndTraining();

    public void Initialize(IInitializer initializer);

    public bool DoesShapeMatchInputShape(Shape3D shape);

    public void Visit(ILayerVisitor visitor);
    public void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args);
    public T Visit<T>(ILayerOutputVisitor<T> visitor);
    public TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args);
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
    public abstract FeatureSet<float> EvaluateSync(FeatureSet<float> channels);

    public virtual BatchedFeatureSet<float> EvaluateSync(BatchedFeatureSet<float> features) {
        var results = new FeatureSet<float>[features.Batches];
        Parallel.For(0, results.Length, i => {
            results[i] = EvaluateSync(features[i]);
        });
        return new BatchedFeatureSet<float>(results);
    }

    public abstract BackpropagationReturns Backpropagate(BackpropagationArgs args);

    public abstract void SubtractGradients(LayerGradients? gradients);

    public abstract void Visit(ILayerVisitor visitor);
    public abstract void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args);
    public abstract T Visit<T>(ILayerOutputVisitor<T> visitor);
    public abstract TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args);
}