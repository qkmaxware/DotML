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
    public double GetParameterUpdate(int timestep, double baseLearningRate, double gradient, int parameterIndex);
}

/// <summary>
/// Static rate optimizer. Keeps the learning rate static across the entire training session.
/// </summary>
public class ConstantRate : ILearningRateOptimizer {
    public void Initialize(int parameterCount) { /* No need to do anything */ }

    public double GetParameterUpdate(int timestep, double baseLearningRate, double gradient, int parameterIndex) {
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
    public double DecayRate {get; init;}

    const double epsilon = 1e-8;

    /// <summary>
    /// Learning rate optimizer based on the RMSProp technique. Learning rate gets smaller as the gradient gets smaller.
    /// </summary>
    /// <param name="decayRate">Decay rate for the learning rate</param>
    public RMSPropOptimizer(double decayRate = 0.9) {
        this.DecayRate = Math.Abs(decayRate); // Can't be -
    }

    private double[] moments = new double[0];

    public void Initialize(int parameters) {
        moments = new double[parameters];
        Array.Fill(moments, 0d);
    }

    /*
        // Update cache with the squared gradient
        cache[index] = decayRate * cache[index] + (1 - decayRate) * gradient * gradient;

        // Compute the adjusted learning rate
        return learningRate / (Math.Sqrt(cache[index]) + epsilon);
    */

    public double GetParameterUpdate(int timestep, double baseLearningRate, double gradient, int parameterIndex) {
        var cached = DecayRate * moments[parameterIndex] + (1 - DecayRate) * gradient * gradient;
        moments[parameterIndex] = cached;

        var denom = Math.Max(cached, epsilon);
        var adjusted_gradient = gradient / Math.Sqrt(denom);
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
    public double Beta1 {get; init;}

    /// <summary>
    /// Second ADAM hyperparameter
    /// </summary>
    public double Beta2 {get; init;}

    const double epsilon = 1e-8;

    struct Moment {
        public double First;
        public double Second;
    }
    private Moment[] moments = new Moment[0];

    public AdamOptimizer(double beta1 = 0.9, double beta2 = 0.999) {
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
    public double GetParameterUpdate(int timestep, double baseLearningRate, double gradient, int parameterIndex) {
        var cached = moments[parameterIndex];

        // Update biased first moment estimate
        cached.First = Beta1 * cached.First + (1 - Beta1) * gradient;

        // Update biased second moment estimate
        cached.Second = Beta2 * cached.Second + (1 - Beta2) * gradient * gradient;

        // Preserve cached value update
        moments[parameterIndex] = cached;

        // Bias correction
        double mHat = cached.First / (1 - Math.Pow(Beta1, timestep));
        double vHat = cached.Second / (1 - Math.Pow(Beta2, timestep));

        // Adjusted learning rate
        vHat = Math.Max(vHat, epsilon);
        var adjusted_gradient = mHat / Math.Sqrt(vHat);
        var parameter_update = baseLearningRate * adjusted_gradient;
        if (double.IsNaN(parameter_update)) {
            throw new ArithmeticException("NaN generated for parameter update.");
        }
        return parameter_update;
    }
    // Dereference of a possibly null reference.
    #pragma warning restore CS8602 
}