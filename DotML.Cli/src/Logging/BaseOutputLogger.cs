using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;

public abstract class BaseOutputLogger : IOutputLogger {

    protected DirectoryInfo LogDirectory {get; private set;}

    public BaseOutputLogger(DirectoryInfo logDir) {
        this.LogDirectory = logDir;
        this.LogDirectory.Create();
    }

    public virtual void Log(string identifier, Tensor<float> output) {}

    public virtual void Visit(ConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args)
    {
        return;
    }

    public virtual void Visit(DepthwiseConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(TransposeConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(PixelShuffle layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(PoolingLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(FlatteningLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(DropoutLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(LayerNorm layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(BatchNorm layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(DenseLinearLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(ActivationLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(SoftmaxLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(InputCapture capture, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(AdditionSkipConnection capture, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public virtual void Visit(ConcatenationSkipConnection capture, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }
}