using System.Drawing;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Network;

/// <summary>
/// Activation layer for a FeedforwardNetwork
/// <see href="https://en.wikipedia.org/wiki/Activation_function"/>
/// </summary>
public class ActivationLayer : FeedforwardNetworkLayer {

    public ActivationFunction ActivationFunction {get; init;}

    public ActivationLayer(Shape3D input_size, ActivationFunction activation) {
        this.InputShape = input_size;
        this.OutputShape = input_size;
        this.ActivationFunction = activation;
    }

    public override void Initialize(IInitializer initializer) { }

    public override int TrainableParameterCount() => 0;

    public override FeatureSet<double> EvaluateSync(FeatureSet<double> channels) {
        var len = channels.Channels;
        Matrix<double>[] outputs = new Matrix<double>[len];
        for (var i = 0; i < len; i++) {
            outputs[i] = channels[i].Transform(ActivationFunction.Invoke);
        }
        return new FeatureSet<double>(outputs);
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        FeatureSet<double>[] batched_input_gradients = new FeatureSet<double>[args.InputBatch.Batches];

        Parallel.For(0, batched_input_gradients.Length, (batchIndex) => {
            var batch = args.InputBatch[batchIndex];

            var input_channels = batch.Channels;
            var input_gradients = new Matrix<double>[input_channels];
            var output_gradients = args.OutputErrors[batchIndex];

            Parallel.For(0, input_channels, channel => {
                var output_gradient = output_gradients[channel];
                var derivative = batch[channel].Transform(ActivationFunction.InvokeDerivative);   // Gradient of vector elements
                var delta = output_gradient.HadamardWith(derivative);                             // Delta of vector elements (column)
                input_gradients[channel] = delta;
            });

            batched_input_gradients[batchIndex] = new FeatureSet<double>(input_gradients);
        });

        // Return the gradients to be propagated to the previous layer
        return new BackpropagationReturns(
            new BatchedFeatureSet<double>(batched_input_gradients),
            null
        );
    }

    public override void SubtractGradients(LayerGradients? gradients) { }

    public override void Visit(ILayerVisitor visitor) => visitor.Visit(this);
    public override void Visit<TIn>(ILayerInputVisitor<TIn> visitor, TIn args) => visitor.Visit(this, args);
    public override T Visit<T>(ILayerOutputVisitor<T> visitor) =>visitor.Visit(this);
    public override TOut Visit<TIn, TOut>(ILayerInputOutputVisitor<TIn, TOut> visitor, TIn args) => visitor.Visit(this, args);

    public override string ToString() {
        return base.ToString() + "(" + ActivationFunction.ToString() + ")";
    }
}