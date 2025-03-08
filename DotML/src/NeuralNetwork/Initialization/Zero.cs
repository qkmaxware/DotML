namespace DotML.Network.Initialization;

/// <summary>
/// An initializer which initializes everything to a constant value
/// </summary>
public class ConstantInitialization: IInitializer {
    private double constant;

    public ConstantInitialization(double constant) {
        this.constant = constant;
    }

    public double RandomBias(int input_count, int output_count, int parameterCount) {
        return constant;
    }

    public double RandomWeight(int input_count, int output_count, int parameterCount) {
        return constant;
    }
}

/// <summary>
/// An initializer which initializes everything to 0
/// </summary>
/// <typeparam name="TNetwork">Network type</typeparam>
public class ZeroInitialization: ConstantInitialization {
    
    public ZeroInitialization(): base(0) { }

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