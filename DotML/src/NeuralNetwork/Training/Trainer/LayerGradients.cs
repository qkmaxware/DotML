using System.Runtime.CompilerServices;

namespace DotML.Network.Training;

public delegate double GradientTransformationHandler(int index, double value);

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
    public abstract void Clip(double weight_threshold, double bias_threshold);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected double ClipValue(double d, double threshold) {
        if (double.IsNaN(d))
            d = 1e-8;
        return Math.Abs(d) > threshold ? Math.Sign(d) * threshold : d;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipVector(Vec<double> vec, double threshold) {
        vec.Apply((value) => ClipValue(value, threshold));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipMatrix(Matrix<double> mat, double threshold) {
        mat.Apply((value) => ClipValue(value, threshold));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipFeatures(FeatureSet<double> features, double threshold) {
        foreach (var matrix in features)
            ClipMatrix(matrix, threshold);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected void ClipBatch(BatchedFeatureSet<double> batch, double threshold) {
        foreach (var features in batch)
            ClipFeatures(features, threshold);
    }
}

/// <summary>
/// Returned values from a backpropagation step of a neural network layer
/// </summary>
public struct BackpropagationReturns {
    public BatchedFeatureSet<double> InputErrors;
    public LayerGradients? Gradients;

    /// <summary>
    /// Alias for InputErrors
    /// </summary>
    public BatchedFeatureSet<double> dX { 
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => InputErrors;
    }

    public BackpropagationReturns(BatchedFeatureSet<double> error, LayerGradients? gradients = null) {
        this.InputErrors = error;
        this.Gradients = gradients;
    }
}

/// <summary>
/// Input values required for backpropagation of neural network layers
/// </summary>
public class BackpropagationArgs {
    public int LayerIndex;
    public BatchedFeatureSet<double> InputBatch;
    public BatchedFeatureSet<double> OutputBatch;
    public BatchedFeatureSet<double> OutputErrors;

    /// <summary>
    /// Alias for InputBatch
    /// </summary>
    public BatchedFeatureSet<double> X {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => InputBatch;
    }
    /// <summary>
    /// Alias for OutputBatch
    /// </summary>
    public BatchedFeatureSet<double> Y {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => OutputBatch;
    }
    /// <summary>
    /// Alias for OutputErrors
    /// </summary>
    public BatchedFeatureSet<double> dY { 
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => OutputErrors;
    }

    public BackpropagationArgs(int layer, BatchedFeatureSet<double> input, BatchedFeatureSet<double> output, BatchedFeatureSet<double> error) {
        this.LayerIndex = layer;
        this.InputBatch = input;
        this.OutputBatch = output;
        this.OutputErrors = error;
    }
}