using System.Runtime.CompilerServices;

namespace DotML.Network.Initialization;

public class NormalXavierInitialization
    : IInitializer
{
    private static Random rng = new Random();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected static float NormalRandom(double mean, double stddev) {
        double u1 = rng.NextDouble();
        double u2 = rng.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return (float)(mean + z0 * stddev);
    }

    public float RandomWeight(int input_count, int output_count, int parameterCount) {
        double stddev = Math.Sqrt(2.0 / (input_count + output_count));
        return NormalRandom(0, stddev);
    }

    public float RandomBias(int input_count, int output_count, int parameterCount) {
        return 0.01f;
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
                neuron.Bias = 0.01f;
            });
        });
    }
}

public class UniformXavierInitialization
    : IInitializer
{
    private static Random rng = new Random();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    protected static float UniformRandom(double limit) {
        return (float)((rng.NextDouble() * 2 * limit) - limit);
    }

    public float RandomWeight(int input_count, int output_count, int parameterCount) {
        double limit = Math.Sqrt(6.0 / (input_count + output_count));
        return UniformRandom(limit);
    }

    public float RandomBias(int input_count, int output_count, int parameterCount) {
        return 0.01f;
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
                neuron.Bias = 0.01f;
            });
        });
    }
}