namespace DotML.Network;

/// <summary>
/// Writer to encode layer weights and biases to a safetensor file
/// </summary>
public class LayerSafetensorWriter : ILayerInputVisitor<int> {

    private Safetensors sb = new Safetensors();

    public Safetensors ToSafetensors() => sb;

    public void Visit(ConvolutionLayer convo, int layerIndex) {
        for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
            var filter = convo.Filters[filterIndex];
            for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                var kernel = filter[kernelIndex];
                sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]", kernel);
            }
            sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Bias", new Matrix<double>(1, 1, filter.Bias));
        }
        return;
    }

    public void Visit(DepthwiseConvolutionLayer convo, int layerIndex) {
        var filter = convo.Filters;
        for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
            var kernel = filter[kernelIndex][0];
            sb.Add($"Layers[{layerIndex}].Filter.Kernel[{kernelIndex}]", kernel);
        }
        sb.Add($"Layers[{layerIndex}].Filter.Bias", new Vec<double>(filter.Select(x => x.Bias).ToArray()));
        return;
    }

    public void Visit(TransposeConvolutionLayer convo, int layerIndex) {
        for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
            var filter = convo.Filters[filterIndex];
            for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                var kernel = filter[kernelIndex];
                sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]", kernel);
            }
            sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Bias", new Matrix<double>(1, 1, filter.Bias));
        }
        return;
    }

    public void Visit(PixelShuffle layer, int layerIndex) { return; }

    public void Visit(PoolingLayer layer, int layerIndex) { return; }

    public void Visit(FlatteningLayer layer, int layerIndex) { return; }

    public void Visit(DropoutLayer layer, int layerIndex) { return; }

    public void Visit(LayerNorm norm, int layerIndex) {
        var gammas = norm.Gammas;
        for (var gammaIndex = 0; gammaIndex < gammas.Length; gammaIndex++) {
            var kernel = gammas[gammaIndex];
            sb.Add($"Layers[{layerIndex}].Gamma[{gammaIndex}]", kernel);
        }
        var betas = norm.Betas;
        for (var betaIndex = 0; betaIndex < betas.Length; betaIndex++) {
            var kernel = betas[betaIndex];
            sb.Add($"Layers[{layerIndex}].Beta[{betaIndex}]", kernel);
        }
        return;
    }

    public void Visit(BatchNorm norm, int layerIndex) {
        sb.Add($"Layers[{layerIndex}].Means", norm.RunningMean);
        sb.Add($"Layers[{layerIndex}].Variance", norm.RunningVariance);
        var gammas = norm.Gammas;
        /*for (var gammaIndex = 0; gammaIndex < gammas.Length; gammaIndex++) {
            var kernel = gammas[gammaIndex];
            sb.Add($"Layers[{layerIndex}].Gamma[{gammaIndex}]", kernel);
        }
        var betas = norm.Betas;
        for (var betaIndex = 0; betaIndex < betas.Length; betaIndex++) {
            var kernel = betas[betaIndex];
            sb.Add($"Layers[{layerIndex}].Beta[{betaIndex}]", kernel);
        }*/
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