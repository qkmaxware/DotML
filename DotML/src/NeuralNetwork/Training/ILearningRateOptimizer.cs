using System.Runtime.CompilerServices;

namespace DotML.Network.Training;

/// <summary>
/// Optimizer to adjust the learning rate during training.
/// </summary>
public interface ILearningRateOptimizer {
    /// <summary>
    /// Initialize the optimizer to work for a given number of parameters
    /// </summary>
    /// <param name="parameterCount">Number of parameters in the network</param>
    public void Initialize(int parameterCount);
    /// <summary>
    /// Initialize the optimizer to work for the trainable parameters of a network
    /// </summary>
    /// <param name="network">Neural network</param>
    public void Initialize(INeuralNetwork network) => Initialize(network.TrainableParameterCount());
    /// <summary>
    /// Return a new gradient which can be used to update a network parameter
    /// </summary>
    /// <param name="baseLearningRate">Base learning rate</param>
    /// <param name="gradient">raw gradient</param>
    /// <param name="timestep">update timestep</param>
    /// <param name="parameterIndex">index of the trainable parameter</param>
    /// <returns>adjusted gradient</returns>
    public float GetParameterUpdate(int timestep, float baseLearningRate, float gradient, int parameterIndex);
}

/// <summary>
/// Container for various learning rate optimizers
/// </summary>
public static class Optimizers {
    /// <summary>
    /// Enumerate over all learning rate optimizers
    /// </summary>
    /// <returns>enumerable of learning rate optimizers</returns>
    public static IEnumerable<ILearningRateOptimizer> EnumerateAll() {
        return typeof(Optimizers)
            .GetProperties(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
            .Where(prop => prop.CanRead && prop.PropertyType.IsAssignableTo(typeof(ILearningRateOptimizer)))
            .Select(prop => prop.GetValue(null))
            .OfType<ILearningRateOptimizer>();
    }

    /// <summary>
    /// Optimizer that maintains a constant learning rate
    /// </summary>
    public static ILearningRateOptimizer ConstantRate {get; private set;} = new ConstantRate();

    /// <summary>
    /// Optimizer that uses RMS prop to adjust the learning rate
    /// </summary>
    public static ILearningRateOptimizer RMSProp {get; private set;} = new RMSPropOptimizer();

    /// <summary>
    /// Optimizer that uses the ADAM technique to adjust the learning rate
    /// </summary>
    public static ILearningRateOptimizer Adam {get; private set;} = new AdamOptimizer();
}

/// <summary>
/// Static rate optimizer. Keeps the learning rate static across the entire training session.
/// </summary>
public class ConstantRate : ILearningRateOptimizer {
    public void Initialize(int parameterCount) { /* No need to do anything */ }

    public float GetParameterUpdate(int timestep, float baseLearningRate, float gradient, int parameterIndex) {
        return baseLearningRate * gradient;
    }
}

/// <summary>
/// Learning rate optimizer based on the RMSProp technique. Learning rate gets smaller as the gradient gets smaller.
/// </summary>
public class RMSPropOptimizer : ILearningRateOptimizer {
    /// <summary>
    /// Decay rate for the learning rate
    /// </summary>
    public float DecayRate {get; init;}

    const float epsilon = 1e-8f;

    /// <summary>
    /// Learning rate optimizer based on the RMSProp technique. Learning rate gets smaller as the gradient gets smaller.
    /// </summary>
    /// <param name="decayRate">Decay rate for the learning rate</param>
    public RMSPropOptimizer(float decayRate = 0.9f) {
        this.DecayRate = MathF.Abs(decayRate); // Can't be -
    }

    private float[] moments = new float[0];

    public void Initialize(int parameters) {
        moments = new float[parameters];
        Array.Fill(moments, 0f);
    }

    /*
        // Update cache with the squared gradient
        cache[index] = decayRate * cache[index] + (1 - decayRate) * gradient * gradient;

        // Compute the adjusted learning rate
        return learningRate / (Math.Sqrt(cache[index]) + epsilon);
    */

    public float GetParameterUpdate(int timestep, float baseLearningRate, float gradient, int parameterIndex) {
        var cached = DecayRate * moments[parameterIndex] + (1 - DecayRate) * gradient * gradient;
        moments[parameterIndex] = cached;

        var denom = Math.Max(cached, epsilon);
        var adjusted_gradient = gradient / MathF.Sqrt(denom);
        var parameter_update = baseLearningRate * adjusted_gradient;

        return parameter_update;
    }
}

/// <summary>
/// Learning rate optimizer based on the ADAM technique. Learning rate gets smaller as the gradient gets smaller.
/// </summary>
public class AdamOptimizer : ILearningRateOptimizer {
    /// <summary>
    /// First ADAM hyperparameter
    /// </summary>
    public float Beta1 {get; init;}

    /// <summary>
    /// Second ADAM hyperparameter
    /// </summary>
    public float Beta2 {get; init;}

    const float epsilon = 1e-8f;

    struct Moment {
        public float First;
        public float Second;
    }
    private Moment[] moments = new Moment[0];

    public AdamOptimizer(float beta1 = 0.9f, float beta2 = 0.999f) {
        this.Beta1 = Math.Abs(beta1);
        this.Beta2 = Math.Abs(beta2);
    }

    public void Initialize(int parameters) {
        moments = new Moment[parameters];
        Array.Fill(moments, new Moment{ First = 0, Second = 0 });
    }

    // TODO double check the below logic. Make NULL safe (I mean shouldnt be an issue since initialize should set everything up... but could be if I forget to call it)
    // Dereference of a possibly null reference.
    #pragma warning disable CS8602
    public float GetParameterUpdate(int timestep, float baseLearningRate, float gradient, int parameterIndex) {
        var cached = moments[parameterIndex];

        // Update biased first moment estimate
        cached.First = Beta1 * cached.First + (1 - Beta1) * gradient;

        // Update biased second moment estimate
        cached.Second = Beta2 * cached.Second + (1 - Beta2) * gradient * gradient;

        // Preserve cached value update
        moments[parameterIndex] = cached;

        // Bias correction
        float mHat = cached.First / (1 - MathF.Pow(Beta1, timestep));
        float vHat = cached.Second / (1 - MathF.Pow(Beta2, timestep));

        // Adjusted learning rate
        vHat = Math.Max(vHat, epsilon);
        var adjusted_gradient = mHat / MathF.Sqrt(vHat);
        var parameter_update = baseLearningRate * adjusted_gradient;
        if (float.IsNaN(parameter_update)) {
            throw new ArithmeticException("NaN generated for parameter update.");
            //parameter_update = 0;
        }
        return parameter_update;
    }
    // Dereference of a possibly null reference.
    #pragma warning restore CS8602 
}