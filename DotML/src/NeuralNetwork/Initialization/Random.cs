namespace DotML.Network.Initialization;

public class RandomInitialization<TNetwork> 
    : IInitializer<TNetwork> 
where TNetwork:ILayeredNeuralNetwork<ILayerWithNeurons> {

    private double min;
    private double max;
    private static Random rng = new Random();

    public RandomInitialization(double min, double max) {
        this.min = Math.Min(min, max);
        this.max = Math.Max(min, max);
    }

    public void InitializeBiases(TNetwork network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                var weights = neuron.Weights;
            
                var weightc = weights.Length;
                for (var i = 0; i < weightc; i++) {
                    var sample = rng.NextDouble();
                    var number = (max * sample) + (min * (1d - sample));
                    weights[i] = number;
                }
            });
        });
    }

    public void InitializeWeights(TNetwork network) {
        network.ForeachLayer(layer => {
            layer.ForeachNeuron(neuron => {
                var weights = neuron.Weights;
                
                var sample = rng.NextDouble();
                var number = (max * sample) + (min * (1d - sample));
                neuron.Bias = number;
            });
        });
    }
}