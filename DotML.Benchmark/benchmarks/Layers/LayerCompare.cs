using BenchmarkDotNet.Attributes;
using CommandLine;
using DotML.Network;

namespace DotML.Benchmark;

[Benchmarkable]
public class CompareLayers {

    [Params(64)]
    public int CHANNELS;
    [Params(32, 64, 128, 256)]
    public int DIM_LENGTH { get; set; }
    [Params(100)]
    public int OUT_CLASSES;

    public struct Args {
        public BatchedFeatureSet<float> output;
        public IFeedforwardNetworkLayer layer;

        public static implicit operator Args(FeedforwardNetworkLayer layer) {
            return new Args {
                layer = layer,
                output = new BatchedFeatureSet<float>(new Shape4D(1, layer.OutputShape.Channels, layer.OutputShape.Rows, layer.OutputShape.Channels))
            };
        }
    }

    private BatchedFeatureSet<float> input;

    private Args act;
    private Args convo;
    private Args tconvo;
    private Args dense;
    private Args dropout;
    private Args flat;
    private Args shuffle;
    private Args maxpool;
    private Args softmax;

    private Args lnorm;

    [GlobalSetup]
    public void Setup() {
        this.input = new BatchedFeatureSet<float>(new Shape4D(1, CHANNELS, DIM_LENGTH, DIM_LENGTH));

        this.act = new ActivationLayer(input[0].Shape, ActivationFunctions.ReLU);
        this.convo = new ConvolutionLayer(input[0].Shape, Padding.Same, filters: ConvolutionFilter.Make(CHANNELS, CHANNELS, 3));
        this.tconvo = new TransposeConvolutionLayer(input[0].Shape, Padding.Same, Expansion.Same, 1, 1, filters: ConvolutionFilter.Make(CHANNELS, CHANNELS, 3));
        this.dense = new DenseLinearLayer(input[0].Shape.Count, OUT_CLASSES);
        this.dropout = new DropoutLayer(input[0].Shape, 0.25f);
        this.flat = new FlatteningLayer(input[0].Shape);
        //this.shuffle;
        this.maxpool = new LocalMaxPoolingLayer(input[0].Shape, 3);
        this.softmax = new SoftmaxLayer(input[0].Shape.Count);

        this.lnorm = new LayerNorm(input[0].Shape);
    }

    [GlobalCleanup]
    public void Takedown() {

    }

    #region Feed Forward

    [Benchmark]
    public void ForwardActivationLayer() {
        act.layer.EvaluateSync(input);
    }
    [Benchmark]
    public void ForwardConvolutionLayer() {
        convo.layer.EvaluateSync(input);
    }
    [Benchmark]
    public void ForwardTransposeConvolutionLayer() {
        tconvo.layer.EvaluateSync(input);
    }
    [Benchmark]
    public void ForwardDenseLinearLayer() {
        dense.layer.EvaluateSync(input);
    }
    [Benchmark]
    public void ForwardDropoutLayer() {
        dropout.layer.EvaluateSync(input);
    }
    [Benchmark]
    public void ForwardFlatteningLayer() {
        flat.layer.EvaluateSync(input);
    }
    [Benchmark]
    public void ForwardMaxPoolLayer() {
        maxpool.layer.EvaluateSync(input);
    }
    [Benchmark]
    public void ForwardSoftmaxLayer() {
        softmax.layer.EvaluateSync(input);
    }
    [Benchmark]
    public void ForwardLayerNormLayer() {
        lnorm.layer.EvaluateSync(input);
    }

    #endregion

    #region Backpropagate
    [Benchmark]
    public void BackwardActivationLayer() {
        var args = act;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    [Benchmark]
    public void BackwardConvolutionLayer() {
        var args = convo;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    [Benchmark]
    public void BackwardTransposeConvolutionLayer() {
        var args = tconvo;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    [Benchmark]
    public void BackwardDenseLinearLayer() {
        var args = dense;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    [Benchmark]
    public void BackwardDropoutLayer() {
        var args = dropout;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    [Benchmark]
    public void BackwardFlatteningLayer() {
        var args = flat;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    [Benchmark]
    public void BackwardMaxPoolLayer() {
        var args = maxpool;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    [Benchmark]
    public void BackwardSoftmaxLayer() {
        var args = softmax;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    [Benchmark]
    public void BackwardLayerNormLayer() {
        var args = lnorm;
        args.layer.Backpropagate(new Network.Training.BackpropagationArgs (
            layer: -1,
            input: input, 
            output: args.output,
            error: args.output
        ));
    }
    #endregion

}