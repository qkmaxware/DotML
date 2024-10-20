namespace DotML.Network.Initialization;

/// <summary>
/// An initializer which doesn't initialize lol
/// </summary>
/// <typeparam name="TNetwork">Network type</typeparam>
public class NoInitialization<TNetwork> : IInitializer<TNetwork> where TNetwork: INeuralNetwork  {
    public static readonly NoInitialization<TNetwork> Instance = new NoInitialization<TNetwork>();
    public void InitializeBiases(TNetwork network) {
        throw new NotImplementedException();
    }

    public void InitializeWeights(TNetwork network) {
        throw new NotImplementedException();
    }
}

/// <summary>
/// An initializer which initializes everything to 0
/// </summary>
/// <typeparam name="TNetwork">Network type</typeparam>
public class ZeroInitialization<TNetwork> : IInitializer<TNetwork> where TNetwork: ILayeredNeuralNetwork  {
    public static readonly NoInitialization<TNetwork> Instance = new NoInitialization<TNetwork>();
    public void InitializeBiases(TNetwork network) {
        network.ForeachNeuron((layer, neuron) => { 
            var weights = neuron.Weights;
            var weightc = weights.Length;

            for (var w = 0; w < weightc; w++) {
                weights[w] = 0;
            }
        });
    }

    public void InitializeWeights(TNetwork network) {
         network.ForeachNeuron((layer, neuron) => { neuron.Bias = 0; });
    }
}