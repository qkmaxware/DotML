namespace DotML.Network.Initialization;

/// <summary>
/// An initializer which initializes everything to 0
/// </summary>
/// <typeparam name="TNetwork">Network type</typeparam>
public class ZeroInitialization: IInitializer {
    
    public double RandomBias(int input_count, int output_count, int parameterCount) {
        return 0;
    }

    public double RandomWeight(int input_count, int output_count, int parameterCount) {
        return 0;
    }

    public void InitializeBiases(ILayeredNeuralNetwork<ILayerWithNeurons> network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                var weights = neuron.Weights;
                var weightc = weights.Length;

                for (var w = 0; w < weightc; w++) {
                    weights[w] = 0;
                }
            });
        });
    }

    public void InitializeWeights(ILayeredNeuralNetwork<ILayerWithNeurons> network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                neuron.Bias = 0; 
            });
        });
    }
}