using System.Runtime.CompilerServices;

namespace DotML.Network.Training;

public delegate float GradientTransformationHandler(int parameterIndex, float parameterValue, float gradient);

/// <summary>
/// Base class for layer gradients
/// </summary>
public abstract class LayerGradients {
    /// <summary>
    /// Apply a transformation for all gradient values
    /// </summary>
    /// <param name="handler">transformation function</param>
    public abstract void Apply(GradientTransformationHandler handler);

    /// <summary>
    /// Clip all gradients
    /// </summary>
    /// <param name="weight_threshold">Clipping threshold for weights</param>
    /// <param name="bias_threshold">Clipping threshold for biases</param>
    public abstract void Clip(float weight_threshold, float bias_threshold);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected float ClipValue(float d, float threshold) {
        if (float.IsNaN(d))
            d = 1e-8f;
        return Math.Abs(d) > threshold ? MathF.Sign(d) * threshold : d;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipVector(Vec<float> vec, float threshold) {
        vec.Apply((value) => ClipValue(value, threshold));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipMatrix(Matrix<float> mat, float threshold) {
        mat.Apply((value) => ClipValue(value, threshold));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipFeatures(FeatureSet<float> features, float threshold) {
        foreach (var matrix in features)
            ClipMatrix(matrix, threshold);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipBatch(BatchedFeatureSet<float> batch, float threshold) {
        foreach (var features in batch)
            ClipFeatures(features, threshold);
    }
}

/// <summary>
/// Returned values from a backpropagation step of a neural network layer
/// </summary>
public struct BackpropagationReturns {
    public BatchedFeatureSet<float> InputErrors;
    public LayerGradients? Gradients;

    /// <summary>
    /// Alias for InputErrors
    /// </summary>
    public BatchedFeatureSet<float> dX { 
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => InputErrors;
    }

    public BackpropagationReturns(BatchedFeatureSet<float> error, LayerGradients? gradients = null) {
        this.InputErrors = error;
        this.Gradients = gradients;
    }
}

/// <summary>
/// Input values required for backpropagation of neural network layers
/// </summary>
public class BackpropagationArgs {
    public int LayerIndex;
    public BatchedFeatureSet<float> InputBatch;
    public BatchedFeatureSet<float> OutputBatch;
    public BatchedFeatureSet<float> OutputErrors;

    /// <summary>
    /// Alias for InputBatch
    /// </summary>
    public BatchedFeatureSet<float> X {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => InputBatch;
    }
    /// <summary>
    /// Alias for OutputBatch
    /// </summary>
    public BatchedFeatureSet<float> Y {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => OutputBatch;
    }
    /// <summary>
    /// Alias for OutputErrors
    /// </summary>
    public BatchedFeatureSet<float> dY { 
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => OutputErrors;
    }

    public BackpropagationArgs(int layer, BatchedFeatureSet<float> input, BatchedFeatureSet<float> output, BatchedFeatureSet<float> error) {
        this.LayerIndex = layer;
        this.InputBatch = input;
        this.OutputBatch = output;
        this.OutputErrors = error;
    }
}