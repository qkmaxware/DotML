using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Layer which performs dropout
/// <see href="https://en.wikipedia.org/wiki/Dilution_(neural_networks)"/>
/// </summary>
[Untested()]
public class DropoutLayer : FeedforwardNetworkLayer {
    public double DropoutRate {get; init;}
    public double KeepRate => 1 - DropoutRate;

    public DropoutLayer(Shape3D input_size) : this(input_size, 0.1) {}
    
    public DropoutLayer(Shape3D input_size, double dropoutRate) {
        this.InputShape = input_size;
        this.OutputShape = input_size;
        this.DropoutRate = Math.Clamp(dropoutRate, 0.0, 1.0);
    }

    public override void Initialize(IInitializer initializer) { }

    protected override void OnTrainingBegin() {
        // Start using a mask at the beginning of a new FF batch
        // This mask is cleared after the end of the batch (including backpropagation)
        this.RegenerateMask();
    }

    protected override void OnTrainingEnd() {
        this.ClearMask();
    }

    public override int TrainableParameterCount() => 0;

    private Random rng = new Random();

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> inputs) { 
        if (this.IsInference)
            return inputs; // No dropout at runtime
        
        // Dropout at training time
        var channelCount = inputs.Channels;
        var outputs = new Matrix<double>[channelCount];
        var mask = this.mask;

        if (channelCount < 1 || mask is null)
            return inputs;

        for (var channel = 0; channel < channelCount; channel++) {
            var input = inputs[channel];
            outputs[channel] = input.HadamardWith(mask[channel]); // Elementwise multiplication with the mask
        }

        return new FeatureSet<double>(outputs);
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        FeatureSet<double>? mask = this.mask;
        if (mask is null) {
            return new BackpropagationReturns(
                args.OutputErrors, // Just pass the errors to the next layer if no mask was assigned
                null
            );
        }

        FeatureSet<double>[] input_errors = new FeatureSet<double>[args.OutputErrors.Batches];
        for (var batchIndex = 0; batchIndex < args.OutputBatch.Batches; batchIndex++) {
            var batch = args.OutputErrors[batchIndex];

            var matrices = new Matrix<double>[batch.Channels];
            for (var channelIndex = 0; channelIndex < batch.Channels; channelIndex++) {
                matrices[channelIndex] = batch[channelIndex].HadamardWith(mask[channelIndex]);
            }
            input_errors[batchIndex] = new FeatureSet<double>(matrices);
        }

        return new BackpropagationReturns(
            new BatchedFeatureSet<double>(input_errors),
            null
        );
    }

    public override void SubtractGradients(LayerGradients? gradients) { }

    private FeatureSet<double>? mask;

    public void ClearMask() {
        this.mask = null;
    }

    public void RegenerateMask() {
        var features = new Matrix<double>[this.InputShape.Channels];
        for (var i = 0; i < this.InputShape.Channels; i++) {
            features[i] = Matrix<double>.Generate(
                this.InputShape.Rows, 
                this.InputShape.Columns, 
                () => rng.NextDouble() < DropoutRate ? 0.0 : 1.0
            );
        }
        this.mask = new FeatureSet<double>(features);
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override T Visit<T>(ILayerVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

}