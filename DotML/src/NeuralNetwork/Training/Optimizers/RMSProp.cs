using System.Numerics;

namespace DotML.Network.Training;

/// <summary>
/// Learning rate optimizer based on the RMSProp technique. Learning rate gets smaller as the gradient gets smaller.
/// </summary>
public class RMSProp : IOptimizer
{
    /// <summary>
    /// Decay rate for the learning rate
    /// </summary>
    public float DecayRate { get; init; }

    /// <summary>
    /// Learning rate optimizer based on the RMSProp technique. Learning rate gets smaller as the gradient gets smaller.
    /// </summary>
    /// <param name="decayRate">Decay rate for the learning rate</param>
    public RMSProp(float decayRate = 0.9f)
    {
        this.DecayRate = MathF.Abs(decayRate); // Can't be -
    }

    private Dictionary<(INetworkModule, string), Tensor<float>> moments = new Dictionary<(INetworkModule, string), Tensor<float>>();

    public void ClearCaches()
    {
        moments.Clear();
    }

    public Tensor<float> GetParameterUpdate(INetworkModule module, string name, float learningRate, Tensor<float> parameter, Tensor<float> gradient)
    {
        // Create the cached moment or get it if it already exists
        if (!moments.TryGetValue((module, name), out var moment))
        {
            moment = Tensor<float>.Zeros(parameter.Shape);
        }

        // if vector is supported and is hardware accelerated use the vectorized version (scalar version used to fill extra un-vectorizable components)
        // otherwise just use scalar version 

        var cached = moment.ElementWiseBinary(
            gradient,
            (mVec, gVec) => DecayRate * mVec + (1 - DecayRate) * gVec * gVec, // Vectorized method
            (m, g) => DecayRate * m + (1 - DecayRate) * g * g  // Scalar (un-vectorized) method
        );
        moments[(module, name)] = cached;

        const float epsilon = 1e-8f;
        var parameter_update = gradient.ElementWiseBinary(
            cached,
            (gradVec, cachedVec) => learningRate * (gradVec / Vector.SquareRoot(Vector.Max(cachedVec, new Vector<float>(epsilon)))), // Vectorized method
            (grad, cached) => learningRate * (grad / MathF.Sqrt(Math.Max(cached, epsilon)))  // Scalar (un-vectorized) method
        );
        if (parameter_update.AsArray().Any((parameter_update) => float.IsNaN(parameter_update)))
        {
            throw new ArithmeticException("NaN generated for parameter update.");
            //parameter_update = 0;
        }
        return parameter_update;
    }
    public void UpdateParameter(INetworkModule module, string name, float learningRate, Tensor<float> parameter, Tensor<float> gradient)
    {
        var update = GetParameterUpdate(module, name, learningRate, parameter, gradient);
        parameter.SubtractWithInplace(update);
    }
}