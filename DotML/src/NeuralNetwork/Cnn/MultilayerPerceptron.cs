namespace DotML.Network;

/// <summary>
/// Utility methods related to the classical Multilayer Perceptron Neural Network architectures
/// </summary>
public static class MultilayerPerceptron {
    /// <summary>
    /// Create a multilayer perception network with the given number of neurons per layer
    /// </summary>
    /// <param name="activation">Nonlinear activation function</param>
    /// <param name="input_size">input size (neurons)</param>
    /// <param name="layer_sizes">neurons for hidden/output layer</param>
    /// <returns>network</returns>
    public static ConvolutionalFeedforwardNetwork Make(ActivationFunction? activation, int input_size, params int[] layer_sizes) {
        if (layer_sizes.Length < 1)
            return new ConvolutionalFeedforwardNetwork();

        // EG 2->2->1 should be a network with 2 actual layer objects the first with (2, 2) the second is (2, 1)

        // First layer (input -> layer[0])
        var ilayer = new FullyConnectedLayer(input_size, layer_sizes[0]);
        var set = new LayerSequencer(ilayer);
        if (activation is not null) {
            set = set.Then((ishape) => new ActivationLayer(ishape, activation));
        }

        // Subsequent layers (layer[i-1] -> layer[i])
        for (var i = 1; i < layer_sizes.Length; i++) {
            set = set.Then((ishape) => new FullyConnectedLayer(ishape.Count, layer_sizes[i]));
            if (activation is not null) {
                set = set.Then((ishape) => new ActivationLayer(ishape, activation));
            }
        }

        return new ConvolutionalFeedforwardNetwork(set);
    }
}