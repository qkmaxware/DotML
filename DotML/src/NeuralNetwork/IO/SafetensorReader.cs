namespace DotML.Network;

/// <summary>
/// Writer to decode layer weights and biases from a safetensor file
/// </summary>
public class LayerSafetensorReader : IConvolutionalLayerVisitor<int, bool> {

    private Safetensors sb;

    public LayerSafetensorReader(Safetensors source) {
        this.sb = source;
    }

    public bool Visit(ConvolutionLayer convo, int layerIndex) {
        for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
            var filter = convo.Filters[filterIndex];
            for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                var key = $"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]";
                if (sb.ContainsKey(key)) {
                    filter[kernelIndex] = sb.GetTensor<double>(key);
                }
            }
            var fbkey = $"Layers[{layerIndex}].Filters[{filterIndex}].Bias";
            if (sb.ContainsKey(fbkey)) {
                filter.Bias = sb.GetTensor<double>(fbkey)[0, 0];
            }
        }
        return true;
    }

    public bool Visit(DepthwiseConvolutionLayer convo, int layerIndex) {
        var filter = convo.Filter;
        for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
            var kernel = filter[kernelIndex];
            var kkey = $"Layers[{layerIndex}].Filter.Kernel[{kernelIndex}]";
            if (sb.ContainsKey(kkey)) {
                filter[kernelIndex] = sb.GetTensor<double>(kkey);
            }
        }
        var bkey = $"Layers[{layerIndex}].Filter.Bias";
        if (sb.ContainsKey(bkey)) {
            filter.Bias = sb.GetTensor<double>(bkey)[0, 0];
        }
        return true;
    }

    public bool Visit(PoolingLayer layer, int layerIndex) { return true; }

    public bool Visit(FlatteningLayer layer, int layerIndex) { return true; }

    public bool Visit(DropoutLayer layer, int layerIndex) { return true; }

    public bool Visit(LayerNorm norm, int layerIndex) {
        var gammas = norm.Gammas;
        for (var gammaIndex = 0; gammaIndex < gammas.Length; gammaIndex++) {
            var key = $"Layers[{layerIndex}].Gamma[{gammaIndex}]";
            if (sb.ContainsKey(key)) {
                gammas[gammaIndex] = sb.GetTensor<double>(key);
            }
        }
        var betas = norm.Betas;
        for (var betaIndex = 0; betaIndex < betas.Length; betaIndex++) {
            var key = $"Layers[{layerIndex}].Beta[{betaIndex}]";
            if (sb.ContainsKey(key)) {
                betas[betaIndex] = sb.GetTensor<double>(key);
            }
        }
        return true;
    }

    public bool Visit(BatchNorm norm, int layerIndex) {
        var gammas = norm.Gammas;
        for (var gammaIndex = 0; gammaIndex < gammas.Length; gammaIndex++) {
            var key = $"Layers[{layerIndex}].Gamma[{gammaIndex}]";
            if (sb.ContainsKey(key)) {
                gammas[gammaIndex] = sb.GetTensor<double>(key);
            }
        }
        var betas = norm.Betas;
        for (var betaIndex = 0; betaIndex < betas.Length; betaIndex++) {
            var key = $"Layers[{layerIndex}].Beta[{betaIndex}]";
            if (sb.ContainsKey(key)) {
                betas[betaIndex] = sb.GetTensor<double>(key);
            }
        }
        return true;
    }

    public bool Visit(FullyConnectedLayer conn, int layerIndex) {
        var wkey = $"Layers[{layerIndex}].Weights";
        if (sb.ContainsKey(wkey)) {
            conn.Weights = sb.GetTensor<double>(wkey);
        }
        var bkey = $"Layers[{layerIndex}].Biases";
        if (sb.ContainsKey(bkey)) {
            conn.Biases = Vec<double>.Wrap(sb.GetTensor<double>(bkey).FlattenRows().ToArray());
        }
        return true;
    }

    public bool Visit(ActivationLayer layer, int layerIndex) { return true; }

    public bool Visit(SoftmaxLayer layer, int layerIndex) { return true; }
}