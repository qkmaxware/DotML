using System.Runtime.CompilerServices;

namespace DotML.Network.Initialization;

public class NormalXavierInitialization
    : IInitializer
{
    private static Random rng = new Random();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected static double NormalRandom(double mean, double stddev) {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + z0 * stddev;
    }

    public double RandomWeight(int parameterCount) {
        double stddev = Math.Sqrt(2.0 / parameterCount);
        return NormalRandom(0, stddev);
    }

    public double RandomBias(int parameterCount) {
        return 0.01;
    }

    public void InitializeWeights(ILayeredNeuralNetwork<ILayerWithNeurons> network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                double stddev = Math.Sqrt(2.0 / (layer.InputShape.Count + layer.OutputShape.Count));
                var weights = neuron.Weights;
                var weightc = weights.Length;

                for (var w = 0; w < weightc; w++) {
                    weights[w] = NormalRandom(0, stddev);
                }
            });
        });
    }

    public void InitializeBiases(ILayeredNeuralNetwork<ILayerWithNeurons> network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                //double stddev = Math.Sqrt(2.0 / (layer.InputCount + layer.OutputCount));
                //neuron.Bias = NormalRandom(0, stddev);
                neuron.Bias = 0.01;
            });
        });
    }
}

public class UniformXavierInitialization
    : IInitializer
{
    private static Random rng = new Random();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected static double UniformRandom(double limit) {
        return (rng.NextDouble() * 2 * limit) - limit;
    }

    public double RandomWeight(int parameterCount) {
        double limit = Math.Sqrt(6.0 / parameterCount);
        return UniformRandom(limit);
    }

    public double RandomBias(int parameterCount) {
        return 0.01;
    }

    public void InitializeWeights(ILayeredNeuralNetwork<ILayerWithNeurons> network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                double limit = Math.Sqrt(6.0 / (layer.InputShape.Count + layer.OutputShape.Count));
                var weights = neuron.Weights;
                var weightc = weights.Length;

                for (var w = 0; w < weightc; w++) {
                    weights[w] = UniformRandom(limit);
                }
            });
        });
    }

    public void InitializeBiases(ILayeredNeuralNetwork<ILayerWithNeurons> network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                //double limit = Math.Sqrt(6.0 / (layer.InputCount + layer.OutputCount));
                //neuron.Bias = UniformRandom(limit);
                neuron.Bias = 0.01;
            });
        });
    }
}