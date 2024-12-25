namespace DotML.Network;

/// <summary>
/// Writer to encode layer weights and biases to a safetensor file
/// </summary>
public class LayerSafetensorWriter : IConvolutionalLayerVisitor<int, bool> {

    private Safetensors sb = new Safetensors();

    public Safetensors ToSafetensors() => sb;

    public bool Visit(ConvolutionLayer convo, int layerIndex) {
        for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
            var filter = convo.Filters[filterIndex];
            for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                var kernel = filter[kernelIndex];
                sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]", kernel);
            }
            sb.Add($"Layers[{layerIndex}].Filters[{filterIndex}].Bias", new Matrix<double>(1, 1, filter.Bias));
        }
        return true;
    }

    public bool Visit(DepthwiseConvolutionLayer convo, int layerIndex) {
        var filter = convo.Filter;
        for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
            var kernel = filter[kernelIndex];
            sb.Add($"Layers[{layerIndex}].Filter.Kernel[{kernelIndex}]", kernel);
        }
        sb.Add($"Layers[{layerIndex}].Filter.Bias", new Matrix<double>(1, 1, filter.Bias));
        return true;
    }

    public bool Visit(PoolingLayer layer, int layerIndex) { return true; }

    public bool Visit(FlatteningLayer layer, int layerIndex) { return true; }

    public bool Visit(DropoutLayer layer, int layerIndex) { return true; }

    public bool Visit(LayerNorm norm, int layerIndex) {
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
        return true;
    }

    public bool Visit(FullyConnectedLayer conn, int layerIndex) {
        sb.Add($"Layers[{layerIndex}].Weights", conn.Weights);
        sb.Add($"Layers[{layerIndex}].Biases", conn.Biases);
        return true;
    }

    public bool Visit(ActivationLayer layer, int layerIndex) { return true; }

    public bool Visit(SoftmaxLayer layer, int layerIndex) { return true; }
}