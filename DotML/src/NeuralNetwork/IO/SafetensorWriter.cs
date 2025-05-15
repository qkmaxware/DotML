namespace DotML.Network;

/// <summary>
/// Writer to encode layer weights and biases to a safetensor file
/// </summary>
public class LayerSafetensorWriter : ILayerInputVisitor<int> {

    private Safetensors sb = new Safetensors();

    public Safetensors ToSafetensors() => sb;

    public void Visit(ConvolutionLayer convo, int layerIndex) {
        sb.Add($"Layers[{layerIndex}].Weights", convo.Weights);
        sb.Add($"Layers[{layerIndex}].Biases", convo.Biases);
        /*for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
            var filter = convo.Filters[filterIndex];
            for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                var kernel = filter[kernelIndex];
                sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]", kernel);
            }
            sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Bias", new Matrix<double>(1, 1, filter.Bias));
        }*/
        return;
    }

    public void Visit(DepthwiseConvolutionLayer convo, int layerIndex) {
        sb.Add($"Layers[{layerIndex}].Weights", convo.Weights);
        sb.Add($"Layers[{layerIndex}].Biases", convo.Biases);
        /*var filter = convo.Filters;
        for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
            var kernel = filter[kernelIndex][0];
            sb.Add($"Layers[{layerIndex}].Filter.Kernel[{kernelIndex}]", kernel);
        }
        sb.Add($"Layers[{layerIndex}].Filter.Bias", new Vec<double>(filter.Select(x => x.Bias).ToArray()));*/
        return;
    }

    public void Visit(TransposeConvolutionLayer convo, int layerIndex) {
        sb.Add($"Layers[{layerIndex}].Weights", convo.Weights);
        sb.Add($"Layers[{layerIndex}].Biases", convo.Biases);
        /*for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
            var filter = convo.Filters[filterIndex];
            for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                var kernel = filter[kernelIndex];
                sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]", kernel);
            }
            sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Bias", new Matrix<double>(1, 1, filter.Bias));
        }*/
        return;
    }

    public void Visit(PixelShuffle layer, int layerIndex) { return; }

    public void Visit(PoolingLayer layer, int layerIndex) { return; }

    public void Visit(FlatteningLayer layer, int layerIndex) { return; }

    public void Visit(DropoutLayer layer, int layerIndex) { return; }

    public void Visit(LayerNorm norm, int layerIndex) {
        var gamma_features = new FeatureSet<double>(norm.Gammas);
        sb.Add($"Layers[{layerIndex}].Gammas", gamma_features);
        var beta_features = new FeatureSet<double>(norm.Betas);
        sb.Add($"Layers[{layerIndex}].Betas", beta_features);
        return;
    }

    public void Visit(BatchNorm norm, int layerIndex) {
        sb.Add($"Layers[{layerIndex}].Means", norm.RunningMean);
        sb.Add($"Layers[{layerIndex}].Variance", norm.RunningVariance);
        var gammas = norm.Gammas;
        return;
    }

    public void Visit(DenseLinearLayer conn, int layerIndex) {
        sb.Add($"Layers[{layerIndex}].Weights", conn.Weights);
        sb.Add($"Layers[{layerIndex}].Biases", conn.Biases);
        return;
    }

    public void Visit(ActivationLayer layer, int layerIndex) { return; }

    public void Visit(SoftmaxLayer layer, int layerIndex) { return; }

    public void Visit(InputCapture capture, int args) { return; }

    public void Visit(AdditionSkipConnection capture, int args) { return; }

    public void Visit(ConcatenationSkipConnection capture, int args) { return; }
}