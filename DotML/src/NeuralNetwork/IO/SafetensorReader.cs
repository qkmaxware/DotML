namespace DotML.Network;

/// <summary>
/// Writer to decode layer weights and biases from a safetensor file
/// </summary>
public class LayerSafetensorReader : ILayerInputVisitor<int> {

    private Safetensors sb;

    public LayerSafetensorReader(Safetensors source) {
        this.sb = source;
    }

    public void Visit(ConvolutionLayer convo, int layerIndex) {
        // Old 2d by 2d method
        for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
            var filter = convo.Filters[filterIndex];
            for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                var key = $"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]";
                if (sb.ContainsKey(key)) {
                    filter[kernelIndex] = sb.GetMatrix<double>(key);
                }
            }
            var fbkey = $"Layers[{layerIndex}].Filters[{filterIndex}].Bias";
            if (sb.ContainsKey(fbkey)) {
                filter.Bias = sb.GetMatrix<double>(fbkey)[0, 0];
            }
        }

        // New generic tensor method
        if (sb.ContainsKey($"Layers[{layerIndex}].Weights")) {
            var weights = sb.GetTensor<double>($"Layers[{layerIndex}].Weights");
            weights.CopyTo(convo.Weights);
        }
        if (sb.ContainsKey($"Layers[{layerIndex}].Biases")) {
            var biases = sb.GetTensor<double>($"Layers[{layerIndex}].Biases");
            biases.CopyTo(convo.Biases);
        }
        return;
    }

    public void Visit(DepthwiseConvolutionLayer convo, int layerIndex) {
        // Old 2d by 2d method
        var filter = convo.Filters;
        for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
            var kernel = filter[kernelIndex][0];
            var kkey = $"Layers[{layerIndex}].Filter.Kernel[{kernelIndex}]";
            if (sb.ContainsKey(kkey)) {
                filter[kernelIndex][0] = sb.GetMatrix<double>(kkey);
            }
        }
        var bkey = $"Layers[{layerIndex}].Filter.Bias";
        if (sb.ContainsKey(bkey)) {
            int index = 0;
            foreach (var bias in sb.GetMatrix<double>(bkey)) {
                filter[index++].Bias = bias;
            }
        }

        // New generic tensor method
        if (sb.ContainsKey($"Layers[{layerIndex}].Weights")) {
            var weights = sb.GetTensor<double>($"Layers[{layerIndex}].Weights");
            weights.CopyTo(convo.Weights);
        }
        if (sb.ContainsKey($"Layers[{layerIndex}].Biases")) {
            var biases = sb.GetTensor<double>($"Layers[{layerIndex}].Biases");
            biases.CopyTo(convo.Biases);
        }
        return;
    }

    public void Visit(TransposeConvolutionLayer convo, int layerIndex) {
        // Old 2d by 2d method
        for (var filterIndex = 0; filterIndex < convo.FilterCount; filterIndex++) {
            var filter = convo.Filters[filterIndex];
            for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                var key = $"Layers[{layerIndex}].Filters[{filterIndex}].Kernel[{kernelIndex}]";
                if (sb.ContainsKey(key)) {
                    filter[kernelIndex] = sb.GetMatrix<double>(key);
                }
            }
            var fbkey = $"Layers[{layerIndex}].Filters[{filterIndex}].Bias";
            if (sb.ContainsKey(fbkey)) {
                filter.Bias = sb.GetMatrix<double>(fbkey)[0, 0];
            }
        }

        // New generic tensor method
        if (sb.ContainsKey($"Layers[{layerIndex}].Weights")) {
            var weights = sb.GetTensor<double>($"Layers[{layerIndex}].Weights");
            weights.CopyTo(convo.Weights);
        }
        if (sb.ContainsKey($"Layers[{layerIndex}].Biases")) {
            var biases = sb.GetTensor<double>($"Layers[{layerIndex}].Biases");
            biases.CopyTo(convo.Biases);
        }
        return;
    }

    public void Visit(PixelShuffle layer, int layerIndex) { return; }

    public void Visit(PoolingLayer layer, int layerIndex) { return; }

    public void Visit(FlatteningLayer layer, int layerIndex) { return; }

    public void Visit(DropoutLayer layer, int layerIndex) { return; }

    public void Visit(LayerNorm norm, int layerIndex) {
        // Old 2d by 2d method
        var gammas = norm.Gammas;
        for (var gammaIndex = 0; gammaIndex < gammas.Length; gammaIndex++) {
            var key = $"Layers[{layerIndex}].Gamma[{gammaIndex}]";
            if (sb.ContainsKey(key)) {
                gammas[gammaIndex] = sb.GetMatrix<double>(key);
            }
        }
        var betas = norm.Betas;
        for (var betaIndex = 0; betaIndex < betas.Length; betaIndex++) {
            var key = $"Layers[{layerIndex}].Beta[{betaIndex}]";
            if (sb.ContainsKey(key)) {
                betas[betaIndex] = sb.GetMatrix<double>(key);
            }
        }

        // New generic tensor method
        if (sb.ContainsKey($"Layers[{layerIndex}].Gammas")) {
            var weights = sb.GetTensor<double>($"Layers[{layerIndex}].Gammas");
            var gamma_features = new FeatureSet<double>(norm.Gammas);
            weights.CopyTo(gamma_features);
        }
        if (sb.ContainsKey($"Layers[{layerIndex}].Betas")) {
            var biases = sb.GetTensor<double>($"Layers[{layerIndex}].Betas");
            var beta_features = new FeatureSet<double>(norm.Betas);
            biases.CopyTo(beta_features);
        }
        return;
    }

    public void Visit(BatchNorm norm, int layerIndex) {
        var mean_key = $"Layers[{layerIndex}].Means";
        if (sb.ContainsKey(mean_key)) {
            norm.RunningMean = Vec<double>.Wrap(sb.GetMatrix<double>(mean_key).FlattenRows().ToArray());
        }
        var variance_key = $"Layers[{layerIndex}].Variance";
        if (sb.ContainsKey(variance_key)) {
            norm.RunningVariance = Vec<double>.Wrap(sb.GetMatrix<double>(variance_key).FlattenRows().ToArray());
        }
        
        /*var gammas = norm.Gammas;
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
        }*/
        return;
    }

    public void Visit(DenseLinearLayer conn, int layerIndex) {
        var wkey = $"Layers[{layerIndex}].Weights";
        if (sb.ContainsKey(wkey)) {
            conn.Weights = sb.GetMatrix<double>(wkey);
        }
        var bkey = $"Layers[{layerIndex}].Biases";
        if (sb.ContainsKey(bkey)) {
            conn.Biases = Vec<double>.Wrap(sb.GetMatrix<double>(bkey).FlattenRows().ToArray());
        }
        return;
    }

    public void Visit(ActivationLayer layer, int layerIndex) { return; }

    public void Visit(SoftmaxLayer layer, int layerIndex) { return; }

    public void Visit(InputCapture capture, int args) { return; }

    public void Visit(AdditionSkipConnection capture, int args) { return; }

    public void Visit(ConcatenationSkipConnection capture, int args) { return; }
}