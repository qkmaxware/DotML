namespace DotML.Network.IO.Netbuild;

/// <summary>
/// Writer to encode network architecture to a NetBuild file
/// </summary>
public class NetbuildLayerEncoder : ILayerVisitor {

    protected TextWriter sb;

    public NetbuildLayerEncoder(TextWriter writer) {
        this.sb = writer;
    }

    public void Encode(FeedforwardNetwork network) {
        // Header
        sb.WriteLine("FROM scratch");
        sb.WriteLine($"INPUT {network.InputShape.Channels} {network.InputShape.Rows} {network.InputShape.Columns}");
        if (network is INamedNetwork named && !string.IsNullOrEmpty(named.Name))
            sb.WriteLine($"NAME \"{named.Name}\"");
        sb.WriteLine();

        // BODY
        for (var layerIndex = 0; layerIndex < network.LayerCount; layerIndex++) {
            var layer = network.GetLayer(layerIndex);
            layer.Visit(this);
        }

        // Footer
    }

    public void Visit(ConvolutionLayer layer) {
        sb.WriteLine($"ADD convolution stride-x={layer.StrideX} stride-y={layer.StrideY} padding={layer.Padding} filters={layer.FilterCount} kernel={layer.Filters.Select(x => Math.Max(x.Width, x.Height)).Max()}");
    }

    public void Visit(DepthwiseConvolutionLayer layer) {
        sb.WriteLine($"ADD depthwise stride-x={layer.StrideX} stride-y={layer.StrideY} padding={layer.Padding} kernel={layer.Filters.Select(x => Math.Max(x.Width, x.Height)).Max()}");
    }

    public void Visit(PoolingLayer layer) {
        switch (layer) {
            case LocalMaxPoolingLayer max:
                sb.WriteLine($"ADD maxpool stride-x={layer.StrideX} stride-y={layer.StrideY} kernel={Math.Max(layer.FilterHeight, layer.FilterWidth)}");
                break;
            case LocalAvgPoolingLayer avg:
                sb.WriteLine($"ADD avgpool stride-x={layer.StrideX} stride-y={layer.StrideY} kernel={Math.Max(layer.FilterHeight, layer.FilterWidth)}");
                break;
            default:
                throw new NotImplementedException();
        }
    }

    public void Visit(FlatteningLayer layer) {
        sb.WriteLine($"ADD flattening");
    }

    public void Visit(DropoutLayer layer) {
        sb.WriteLine($"ADD dropout percent={layer.DropoutRate}");
    }

    public void Visit(LayerNorm layer) {
        sb.WriteLine($"ADD layernorm");
    }

    public void Visit(BatchNorm layer) {
        sb.WriteLine($"ADD batchnorm");
    }

    public void Visit(DenseLinearLayer layer) {
        sb.WriteLine($"ADD dense neurons={layer.NeuronCount}");
    }

    public void Visit(ActivationLayer layer) {
        var alpha = layer.ActivationFunction.GetType().GetProperty("Alpha")?.GetValue(layer.ActivationFunction);
        if (alpha is null) {
            sb.WriteLine($"ADD activation fn={layer.ActivationFunction.GetType().Name}");
        } else {
            sb.WriteLine($"ADD activation fn={layer.ActivationFunction.GetType().Name} alpha={alpha}");
        }
    }

    public void Visit(SoftmaxLayer layer) {
        sb.WriteLine($"ADD softmax");
    }

    public void Visit(InputCapture capture) {
        throw new NotImplementedException();
    }

}