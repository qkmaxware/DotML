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

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> channels) {
        var len = channels.Channels;
        Matrix<float>[] outputs = new Matrix<float>[len];
        for (var i = 0; i < len; i++) {
            outputs[i] = channels[i].Transform(ActivationFunction.Invoke);
        }
        return new FeatureSet<float>(outputs);
    }

    public override BackpropagationReturns Backpropagate(BackpropagationArgs args) {
        FeatureSet<float>[] batched_input_gradients = new FeatureSet<float>[args.InputBatch.Batches];

        Parallel.For(0, batched_input_gradients.Length, (batchIndex) => {
            var batch = args.InputBatch[batchIndex];

            var input_channels = batch.Channels;
            var input_gradients = new Matrix<float>[input_channels];
            var output_gradients = args.OutputErrors[batchIndex];

            Parallel.For(0, input_channels, channel => {
                var output_gradient = output_gradients[channel];
                var derivative = batch[channel].Transform(ActivationFunction.InvokeDerivative);   // Gradient of vector elements
                derivative.HadamardWithInplace(output_gradient);
                input_gradients[channel] = derivative;
            });

            batched_input_gradients[batchIndex] = new FeatureSet<float>(input_gradients);
        });

        // Return the gradients to be propagated to the previous layer
        return new BackpropagationReturns(
            new BatchedFeatureSet<float>(batched_input_gradients),
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