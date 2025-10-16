using System.Numerics;

namespace DotML.Network.Training;

/// <summary>
/// Learning rate optimizer based on the ADAM technique with an additional weight decay. Learning rate gets smaller as the gradient gets smaller.
/// </summary>
public class AdamW : IOptimizer
{
    /// <summary>
    /// First ADAM hyperparameter
    /// </summary>
    public float Beta1 { get; init; }

    /// <summary>
    /// Second ADAM hyperparameter
    /// </summary>
    public float Beta2 { get; init; }

    /// <summary>
    /// Decoupled weight decay rate (AdamW-specific)
    /// </summary>
    public float WeightDecay { get; init; }

    struct Moment
    {
        public int Timestep;
        public Tensor<float> First;
        public Tensor<float> Second;
    }

    public AdamW(float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0.01f)
    {
        this.Beta1 = Math.Abs(beta1);
        this.Beta2 = Math.Abs(beta2);
        this.WeightDecay = Math.Max(0f, weightDecay); // Clamp to non-negative
    }

    private Dictionary<(INetworkModule, string), Moment> moments = new Dictionary<(INetworkModule, string), Moment>();

    public void ClearCaches()
    {
        moments.Clear();
    }

    public Tensor<float> GetParameterUpdate(INetworkModule module, string name, float learningRate, Tensor<float> parameter, Tensor<float> gradient)
    {
        // Create the cached moment or get it if it already exists
        if (!moments.TryGetValue((module, name), out var moment))
        {
            moment = new Moment
            {
                Timestep = 1, // Timestep defaults to 1
                First = Tensor<float>.Zeros(parameter.Shape),
                Second = Tensor<float>.Zeros(parameter.Shape)
            };
        }
        else
        {
            moment.Timestep++; // If it already exists, increment timestep
        }

        // NOTE, all Vectorized and Scalar functions should be "functionally" equivalent

        // Update biased first moment estimate
        moment.First = moment.First.ElementWiseBinary(
            gradient,
            (mVec, gVec) => Beta1 * mVec + (1 - Beta1) * gVec, // Vectorized function
            (m, g) => Beta1 * m + (1 - Beta1) * g               // Scalar function
        );

        // Update biased second moment estimate
        moment.Second = moment.Second.ElementWiseBinary(
            gradient,
            (mVec, gVec) => Beta2 * mVec + (1 - Beta2) * gVec * gVec, // Vectorized function
            (m, g) => Beta2 * m + (1 - Beta2) * g * g               // Scalar function
        );

        // Preserve cached value update
        moments[(module, name)] = moment;

        // Bias correction
        Tensor<float> mHat = moment.First / (1 - MathF.Pow(Beta1, moment.Timestep));
        Tensor<float> vHat = moment.Second / (1 - MathF.Pow(Beta2, moment.Timestep));

        // Adjusted learning rate
        const float epsilon = 1e-8f;
        var adam_update = mHat.ElementWiseBinary(
            vHat,
            // Vectorized function
            (mVec, vVec) => learningRate * (mVec / Vector.SquareRoot(Vector.Max(vVec, new Vector<float>(epsilon)))),
            // Scalar function
            (m, v) => learningRate * (m / MathF.Sqrt(Math.Max(v, epsilon)))
        );

        adam_update.ElementWiseBinaryInplace(parameter, (adam, para) => adam + para * learningRate * WeightDecay);

        if (adam_update.AsArray().Any((parameter_update) => float.IsNaN(parameter_update)))
        {
            throw new ArithmeticException("NaN generated for parameter update.");
            //parameter_update = 0;
        }
        return adam_update;
    }

    public void UpdateParameter(INetworkModule module, string name, float learningRate, Tensor<float> parameter, Tensor<float> gradient)
    {
        var update = GetParameterUpdate(module, name, learningRate, parameter, gradient);
        parameter.SubtractWithInplace(update);
    }
}