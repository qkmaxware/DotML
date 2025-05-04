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
            sb.WriteLine($"LABEL \"{named.Name}\"");
        sb.WriteLine();

        // BODY
        for (var layerIndex = 0; layerIndex < network.LayerCount; layerIndex++) {
            var layer = network.GetLayer(layerIndex);
            layer.Visit(this);
        }

        // Footer
    }

    public void Visit(ConvolutionLayer layer) {
        sb.WriteLine($"ADD {nameof(ConvolutionLayer)} stride-x={layer.StrideX} stride-y={layer.StrideY} padding-x={layer.ColumnsPadding} padding-y={layer.RowsPadding} filters={layer.FilterCount} kernel={layer.Filters.Select(x => Math.Max(x.Width, x.Height)).Max()}");
    }

    public void Visit(DepthwiseConvolutionLayer layer) {
        sb.WriteLine($"ADD {nameof(DepthwiseConvolutionLayer)} stride-x={layer.StrideX} stride-y={layer.StrideY} padding={layer.Padding} kernel={layer.Filters.Select(x => Math.Max(x.Width, x.Height)).Max()}");
    }

    public void Visit(TransposeConvolutionLayer layer) {
        sb.WriteLine($"ADD {nameof(TransposeConvolutionLayer)} stride-x={layer.StrideX} stride-y={layer.StrideY} padding-x={layer.InputColumnsPadding} padding-y={layer.InputRowsPadding} expand-x={layer.OutputColumnsPadding} expand-y={layer.OutputRowsPadding} filters={layer.FilterCount} kernel={layer.Filters.Select(x => Math.Max(x.Width, x.Height)).Max()}");
    }

    public void Visit(PixelShuffle layer) {
        sb.WriteLine($"ADD {nameof(PixelShuffle)} upscale-by={layer.UpscalingFactor}");
    }

    public void Visit(PoolingLayer layer) {
        switch (layer) {
            case LocalMaxPoolingLayer max:
                sb.WriteLine($"ADD {nameof(LocalMaxPoolingLayer)} stride-x={layer.StrideX} stride-y={layer.StrideY} padding-x={layer.PaddingX} padding-y={layer.PaddingY} kernel={Math.Max(layer.FilterHeight, layer.FilterWidth)}");
                break;
            case LocalAvgPoolingLayer avg:
                sb.WriteLine($"ADD {nameof(LocalAvgPoolingLayer)} stride-x={layer.StrideX} stride-y={layer.StrideY} padding-x={layer.PaddingX} padding-y={layer.PaddingY} kernel={Math.Max(layer.FilterHeight, layer.FilterWidth)}");
                break;
            default:
                throw new NotImplementedException();
        }
    }

    public void Visit(FlatteningLayer layer) {
        sb.WriteLine($"ADD {nameof(FlatteningLayer)}");
    }

    public void Visit(DropoutLayer layer) {
        sb.WriteLine($"ADD {nameof(DropoutLayer)} percent={layer.DropoutRate}");
    }

    public void Visit(LayerNorm layer) {
        sb.WriteLine($"ADD {nameof(LayerNorm)}");
    }

    public void Visit(BatchNorm layer) {
        sb.WriteLine($"ADD {nameof(BatchNorm)}");
    }

    public void Visit(DenseLinearLayer layer) {
        sb.WriteLine($"ADD {nameof(DenseLinearLayer)} neurons={layer.NeuronCount}");
    }

    public void Visit(ActivationLayer layer) {
        var alpha = layer.ActivationFunction.GetType().GetProperty("Alpha")?.GetValue(layer.ActivationFunction);
        if (alpha is null) {
            sb.WriteLine($"ADD {nameof(ActivationLayer)} fn={layer.ActivationFunction.GetType().Name}");
        } else {
            sb.WriteLine($"ADD {nameof(ActivationLayer)} fn={layer.ActivationFunction.GetType().Name} alpha={alpha}");
        }
    }

    public void Visit(SoftmaxLayer layer) {
        sb.WriteLine($"ADD {nameof(SoftmaxLayer)}");
    }

    public void Visit(InputCapture capture) {
        sb.WriteLine($"ADD {nameof(InputCapture)} AS output_{capture.UID()}");
        //sb.WriteLine($"COPY AS out_{capture.UID()}");
    }

    public void Visit(AdditionSkipConnection skip) {
        sb.WriteLine($"ADD {nameof(AdditionSkipConnection)} residual=output_{skip.CaptureSource.UID()}");
        //sb.WriteLine($"RESIDUAL ADDITION {skip.CaptureSource.UID()}");
    }

    public void Visit(ConcatenationSkipConnection skip) {
        sb.WriteLine($"ADD {nameof(ConcatenationSkipConnection)} residual=output_{skip.CaptureSource.UID()} side={skip.ConcatenationSide}");
        //sb.WriteLine($"RESIDUAL ADDITION {skip.CaptureSource.UID()}");
    }

}