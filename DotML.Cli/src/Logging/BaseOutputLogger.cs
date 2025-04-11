using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;

public abstract class BaseOutputLogger : IOutputLogger {

    protected DirectoryInfo LogDirectory {get; private set;}

    public BaseOutputLogger(DirectoryInfo logDir) {
        this.LogDirectory = logDir;
        this.LogDirectory.Create();
    }


    public virtual Void Visit(ConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(DepthwiseConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(PoolingLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(FlatteningLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(DropoutLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(LayerNorm layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(BatchNorm layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(DenseLinearLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(ActivationLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(SoftmaxLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }

    public virtual Void Visit(InputCapture capture, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return Void.Instance;
    }
}