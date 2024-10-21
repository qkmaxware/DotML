using System.Runtime.CompilerServices;

namespace DotML.Network.Initialization;

public class NormalXavierInitialization<TNetwork> 
    : IInitializer<TNetwork> 
where TNetwork:ILayeredNeuralNetwork<ILayerWithNeurons> {
    private static Random rng = new Random();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected static double NormalRandom(double mean, double stddev) {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + z0 * stddev;
    }

    public void InitializeWeights(TNetwork network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                double stddev = Math.Sqrt(2.0 / (layer.InputCount + layer.OutputCount));
                var weights = neuron.Weights;
                var weightc = weights.Length;

                for (var w = 0; w < weightc; w++) {
                    weights[w] = NormalRandom(0, stddev);
                }
            });
        });
    }

    public void InitializeBiases(TNetwork network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                //double stddev = Math.Sqrt(2.0 / (layer.InputCount + layer.OutputCount));
                //neuron.Bias = NormalRandom(0, stddev);
                neuron.Bias = 0.01;
            });
        });
    }
}

public class UniformXavierInitialization<TNetwork> 
    : IInitializer<TNetwork> 
where TNetwork:ILayeredNeuralNetwork<ILayerWithNeurons> {
    private static Random rng = new Random();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected static double UniformRandom(double limit) {
        return (rng.NextDouble() * 2 * limit) - limit;
    }

    public void InitializeWeights(TNetwork network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                double limit = Math.Sqrt(6.0 / (layer.InputCount + layer.OutputCount));
                var weights = neuron.Weights;
                var weightc = weights.Length;

                for (var w = 0; w < weightc; w++) {
                    weights[w] = UniformRandom(limit);
                }
            });
        });
    }

    public void InitializeBiases(TNetwork network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                //double limit = Math.Sqrt(6.0 / (layer.InputCount + layer.OutputCount));
                //neuron.Bias = UniformRandom(limit);
                neuron.Bias = 0.01;
            });
        });
    }
}