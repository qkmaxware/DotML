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
    public float DropoutRate {get; init;}
    public float KeepRate => 1 - DropoutRate;

    public DropoutLayer(Shape3D input_size) : this(input_size, 0.1f) {}
    
    public DropoutLayer(Shape3D input_size, float dropoutRate) {
        this.InputShape = input_size;
        this.OutputShape = input_size;
        this.DropoutRate = Math.Clamp(dropoutRate, 0.0f, 1.0f);
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

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> inputs) { 
        if (this.IsInference)
            return inputs; // No dropout at runtime
        
        // Dropout at training time
        var channelCount = inputs.Channels;
        var outputs = new Matrix<float>[channelCount];
        var mask = this.mask;

        if (channelCount < 1 || mask is null)
            return inputs;

        for (var channel = 0; channel < channelCount; channel++) {
            var input = inputs[channel];
            outputs[channel] = input.HadamardWith(mask[channel]); // Elementwise multiplication with the mask
        }

        return new FeatureSet<float>(outputs);
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        FeatureSet<float>? mask = this.mask;
        if (mask is null) {
            return new BackpropagationReturns(
                args.OutputErrors, // Just pass the errors to the next layer if no mask was assigned
                null
            );
        }

        FeatureSet<float>[] input_errors = new FeatureSet<float>[args.OutputErrors.Batches];
        for (var batchIndex = 0; batchIndex < args.OutputBatch.Batches; batchIndex++) {
            var batch = args.OutputErrors[batchIndex];

            var matrices = new Matrix<float>[batch.Channels];
            for (var channelIndex = 0; channelIndex < batch.Channels; channelIndex++) {
                matrices[channelIndex] = batch[channelIndex].HadamardWith(mask[channelIndex]);
            }
            input_errors[batchIndex] = new FeatureSet<float>(matrices);
        }

        return new BackpropagationReturns(
            new BatchedFeatureSet<float>(input_errors),
            null
        );
    }

    public override void SubtractGradients(LayerGradients? gradients) { }

    private FeatureSet<float>? mask;

    public void ClearMask() {
        this.mask = null;
    }

    public void RegenerateMask() {
        var features = new Matrix<float>[this.InputShape.Channels];
        for (var i = 0; i < this.InputShape.Channels; i++) {
            features[i] = Matrix<float>.Generate(
                this.InputShape.Rows, 
                this.InputShape.Columns, 
                () => rng.NextDouble() < DropoutRate ? 0.0f : 1.0f
            );
        }
        this.mask = new FeatureSet<float>(features);
    }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);
    public override T Visit<T>(ILayerOutputVisitor<T> visitor) => visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

}