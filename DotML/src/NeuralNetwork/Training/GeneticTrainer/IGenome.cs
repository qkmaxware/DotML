using System.Collections;

namespace DotML.Network.Training;

/// <summary>
/// A genome representation for something that can be evolved via genetic algorithms
/// </summary>
public interface IGenome
{
    /// <summary>
    /// Perform a random mutation of this genome
    /// </summary>
    /// <param name="mutationRate">mutation rate</param>
    /// <returns>mutated genome</returns>
    public IGenome Mutate(float mutationRate);

    /// <summary>
    /// Perform a crossover between this genome and another
    /// </summary>
    /// <param name="other">other genome</param>
    /// <param name="mutationRate">mutation rate</param>
    /// <returns>2 children containing opposite genes from each parent</returns>
    public (IGenome First, IGenome  Second) Crossover(IGenome other, float mutationRate);

    /// <summary>
    /// Compute the similarity between this genome and another. Larger values indicate more dissimilarity than smaller values. A value of 0 should mean that the two genomes are identical. This can also be considered a "distance" between this genome and the other.
    /// </summary>
    /// <param name="to">genome to compare against</param>
    /// <returns>similarity score (larger means more dissimilar)</returns>
    public float Dissimilarity(IGenome to);
}

/// <summary>
/// A basic genome implementation for storing a collection of weights/biases across multiple layers
/// </summary>
public class WeightsAndBiasGenome: IGenome
{
    private List<Tensor<float>> weights;
    private List<Tensor<float>> biases;

    public WeightsAndBiasGenome(List<Tensor<float>> weights, List<Tensor<float>> biases)
    {
        this.weights = weights;
        this.biases = biases;
    }

    public int WeightCount => weights.Count;

    public Tensor<float> GetWeight(int weight) => this.weights[weight];

    public int BiasCount => biases.Count;

    public Tensor<float> GetBias(int bias) => this.biases[bias];

    public IGenome Mutate(float mutationRate)
    {
        List<Tensor<float>> newWeights = new List<Tensor<float>>(this.weights.Count);
        List<Tensor<float>> newBiases = new List<Tensor<float>>(this.biases.Count);

        var distribution = Distributions.Normal<float>(mean: 0, stdDev: mutationRate);
        var negMutationRate = mutationRate;

        for (var i = 0; i < this.weights.Count; i++)
        {
            newWeights.Add(weights[i].ElementWise(x => x + Math.Clamp(distribution.Sample(), negMutationRate, mutationRate)));
        }

        for (var i = 0; i < this.biases.Count; i++)
        {
            newBiases.Add(biases[i].ElementWise(x => x + Math.Clamp(distribution.Sample(), negMutationRate, mutationRate)));
        }

        return new WeightsAndBiasGenome(newWeights, newBiases);
    }

    public (IGenome First, IGenome Second) Crossover(IGenome other, float mutationRate)
    {
        if (other is not WeightsAndBiasGenome otherGenome)
            throw new NotSupportedException(other.GetType().Name);

        var rng = Random.Shared;

        List<Tensor<float>> aw = new List<Tensor<float>>(this.weights.Count);
        List<Tensor<float>> ab = new List<Tensor<float>>(this.biases.Count);
        List<Tensor<float>> bw = new List<Tensor<float>>(this.weights.Count);
        List<Tensor<float>> bb = new List<Tensor<float>>(this.biases.Count);

        for (var i = 0; i < Math.Min(this.weights.Count, otherGenome.weights.Count); i++)
        {
            var alpha = (float)rng.NextDouble();
            var beta = 1-alpha;

            aw.Add(alpha * this.weights[i] + beta*otherGenome.weights[i]);
            ab.Add(alpha * this.biases[i] + beta*otherGenome.biases[i]);

            bw.Add(beta * this.weights[i] + alpha*otherGenome.weights[i]);
            bb.Add(beta * this.biases[i] + alpha*otherGenome.biases[i]);
        }

        var distribution = Distributions.Normal<float>(mean: 0, stdDev: mutationRate);
        var negMutationRate = mutationRate;

        foreach (var tensor in aw)
           tensor.ElementWiseInplace(x => x + Math.Clamp(distribution.Sample(), negMutationRate, mutationRate));
        foreach (var tensor in ab)
           tensor.ElementWiseInplace(x => x + Math.Clamp(distribution.Sample(), negMutationRate, mutationRate));
        foreach (var tensor in bw)
           tensor.ElementWiseInplace(x => x + Math.Clamp(distribution.Sample(), negMutationRate, mutationRate));
        foreach (var tensor in bb)
           tensor.ElementWiseInplace(x => x + Math.Clamp(distribution.Sample(), negMutationRate, mutationRate));

        return (new WeightsAndBiasGenome(aw, ab), new WeightsAndBiasGenome(bw, bb));
    }

    public float Dissimilarity(IGenome to)
    {
        if (to is not WeightsAndBiasGenome otherGenome)
            throw new NotSupportedException(to.GetType().Name);

        float distance = 0;
        for (var i = 0; i < Math.Min(this.weights.Count, otherGenome.weights.Count); i++)
        {
            var x = this.weights[i];
            var y = otherGenome.weights[i];

            for (var j = 0; j < Math.Min(x.ElementCount, y.ElementCount); j++) {
                //distance += Math.Abs(y[j] - x[j]);            // L1 distance
                distance += (y[j] - x[j]) * (y[j] - x[j]);      // L2 distance
            }
        }
        return distance;
    }
    
}
