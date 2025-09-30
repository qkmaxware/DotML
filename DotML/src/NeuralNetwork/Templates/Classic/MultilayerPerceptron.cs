namespace DotML.Network.Templates;

/// <summary>
/// Utility methods related to the classical Multilayer Perceptron Neural Network architectures
/// </summary>
public class MultilayerPerceptronFactory : INetworkModuleFactory<MultilayerPerceptronFactory.BuildSettings>
{
    public class BuildSettings
    {
        public ActivationFunction? ActivationFunction { get; set; }
        public int InputSize { get; set; }
        public int[] LayerSizes { get; set; }

        public BuildSettings()
        {
            ActivationFunction = ActivationFunctions.Sigmoid;
            InputSize = 2;
            LayerSizes = new int[] { 2, 1 };
        }
        
        public BuildSettings(ActivationFunction? activationFunction, int inputSize, params int[] layerSizes)
        {
            ActivationFunction = activationFunction;
            InputSize = inputSize;
            LayerSizes = layerSizes;
        }
    }

    /// <summary>
    /// Create a default 2-2-1 multilayer perceptron with sigmoid activations
    /// </summary>
    /// <returns>network</returns>
    public INetworkModule MakeDefault() => Make(new BuildSettings());

    /// <summary>
    /// Make a multilayer perceptron with the given activation function, input size, and layer sizes. This is a convenience wrapper around Make(BuildSettings).
    /// </summary>
    /// <param name="activation">activation function</param>
    /// <param name="input_size">input layer size</param>
    /// <param name="layer_sizes">hidden to output layer sizes</param>
    /// <returns>network</returns>
    public INetworkModule Make(ActivationFunction? activation, int input_size, params int[] layer_sizes) => Make(new BuildSettings(activation, input_size, layer_sizes));

    /// <summary>
    /// Make a multilayer perceptron with the given settings
    /// </summary>
    /// <param name="settings">settings pertaining to activation function, input size, and layer sizes</param>
    /// <returns>network</returns>
    public INetworkModule Make(BuildSettings settings)
    {
        ActivationFunction? activation = settings.ActivationFunction;
        int input_size = settings.InputSize;
        int[] layer_sizes = settings.LayerSizes;

        if (layer_sizes.Length < 1)
            return new SequentialBlock();

        // EG 2->2->1 should be a network with 2 actual layer objects the first with (2, 2) the second is (2, 1)

        // First layer (input -> layer[0])
        var block = new SequentialBlock();
        var ilayer = new DenseLinear(input_size, layer_sizes[0]);
        block.Add(ilayer);
        if (activation is not null)
        {
            block.Add(new ActivationLayer2(activation));
        }

        // Subsequent layers (layer[i-1] -> layer[i])
        for (var i = 1; i < layer_sizes.Length; i++)
        {
            var in_size = layer_sizes[i - 1];
            var out_size = layer_sizes[i];
            var layer = new DenseLinear(in_size, out_size);
            block.Add(layer);
            // Only add activations on middle layers, never last layer (last layer is logits)
            if (i != layer_sizes.Length - 1 && activation is not null)
            {
                block.Add(new ActivationLayer2(activation));
            }
        }

        return block;
    }
}