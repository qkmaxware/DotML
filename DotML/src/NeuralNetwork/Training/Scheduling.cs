using System.Numerics;

namespace DotML.Network.Training;

public interface ILearningRateScheduler
{
    /// <summary>
    /// Compute the learning rate for the given epoch from the starting/base rate
    /// </summary>
    /// <param name="baseRate">starting learning rate</param>
    /// <param name="epochFor">current epoch (o-index)</param>
    /// <returns>learning rate for this given epoch</returns>
    public float RateForEpoch(int epochFor, float baseRate);

    /// <summary>
    /// Step the scheduler at the end of the epoch
    /// </summary>
    /// <param name="epochFor">epoch that just ended</param>
    /// <param name="loss">loss computed for the epoch</param>
    public void Step(int epochFor, Metric<float> loss) {}
}

public class ConstantRate : ILearningRateScheduler
{
    public float RateForEpoch(int epochFor, float baseRate) => baseRate;
}

public class ExponentialDecay : ILearningRateScheduler
{
    public float Decay { get; private set; }
    public int StepInterval { get; private set; }
    public float MinRate { get; private set; }

    public ExponentialDecay(float decay, int every = 1, float minRate = 1e-6f)
    {
        this.Decay = Math.Abs(decay);
        this.StepInterval = Math.Max(1, every);
        this.MinRate = Math.Abs(minRate);
    }

    public static ExponentialDecay HalfEvery(int epochs) => new ExponentialDecay(MathF.Log(2) / epochs);

    public float RateForEpoch(int epochFor, float baseRate)
    {
        var steps = epochFor / StepInterval;
        var rate = baseRate * MathF.Exp(-Decay * steps);
        return MathF.Max(rate, this.MinRate);
    }
}

public class StepDecay : ILearningRateScheduler
{

    public float Gamma { get; private set; }
    public int StepInterval { get; private set; }

    public StepDecay(float gamma, int step)
    {
        this.Gamma = Math.Abs(gamma);
        this.StepInterval = Math.Max(1, step);
    }

    public float RateForEpoch(int epochFor, float baseRate)
    {
        return baseRate * MathF.Pow(Gamma, epochFor / StepInterval);
    }
}

public class PolynomialDecay : ILearningRateScheduler
{
    public float Power { get; private set; }
    public int MaxEpochs { get; private set; }

    public PolynomialDecay(float power, int epochs)
    {
        this.Power = Math.Abs(power);
        this.MaxEpochs = Math.Max(1, epochs);
    }

    public float RateForEpoch(int epochFor, float baseRate)
    {
        return baseRate * MathF.Pow(1 - MathF.Min(epochFor, MaxEpochs) / MaxEpochs, Power);
    }
}

public class CosineAnnealing : ILearningRateScheduler
{
    public int MaxEpochs { get; private set; }

    public CosineAnnealing(int epochs)
    {
        this.MaxEpochs = Math.Max(1, epochs);
    }

    public float RateForEpoch(int epochFor, float baseRate)
    {
        return baseRate * 0.5f * (1 + MathF.Cos(MathF.PI * Math.Min(epochFor, MaxEpochs) / MaxEpochs));
    }
}

/// <summary>
/// Hold a constant value before starting scheduler
/// </summary>
public class ConstantRateWarmup : ILearningRateScheduler
{
    public int WarmupEpochs { get; init; }
    public ILearningRateScheduler Scheduler { get; init; }

    public ConstantRateWarmup(int warmupEpochs, ILearningRateScheduler scheduler)
    {
        this.WarmupEpochs = Math.Max(1, warmupEpochs);
        this.Scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
    }

    public float RateForEpoch(int epochFor, float baseRate)
    {
        if (epochFor < WarmupEpochs)
            return baseRate;                                                // We are in warmup, do nothing
        return Scheduler.RateForEpoch(epochFor - WarmupEpochs, baseRate);   // Out of warmup, delegate to scheduler
    }

    public void Step(int epoch, Metric<float> loss) => Scheduler.Step(epoch, loss);
}

/// <summary>
/// Linearly increase from 0 to base-rate before starting the scheduler
/// </summary>
public class RampUpWarmup : ILearningRateScheduler
{
    public int WarmupEpochs { get; init; }
    public ILearningRateScheduler Scheduler { get; init; }

    public RampUpWarmup(int warmupEpochs, ILearningRateScheduler scheduler)
    {
        this.WarmupEpochs = Math.Max(1, warmupEpochs);
        this.Scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
    }

    public float RateForEpoch(int epochFor, float baseRate)
    {
        if (epochFor < WarmupEpochs)
            return MathF.Min(baseRate, baseRate * (epochFor + 1) / WarmupEpochs); // We are in warmup, ramp up
        return Scheduler.RateForEpoch(epochFor - WarmupEpochs, baseRate);   // Out of warmup, delegate to scheduler
    }

    public void Step(int epoch, Metric<float> loss) => Scheduler.Step(epoch, loss);
}

/// <summary>
/// A scheduler that reduces learning rate when progress has plateaued and restores learning rate when progress resumes
/// </summary>
public class ReduceLROnPlateau : ILearningRateScheduler
{
    /// <summary>
    /// Factor to multiply learning rate by when reducing the learning rate on plateaus
    /// </summary>
    public float ReductionFactor { get; init; }
    /// <summary>
    /// Factor to multiply learning rate by when restoring the learning rate after plateaus
    /// </summary>
    public float RecoveryFactor { get; init; }
    /// <summary>
    /// Number of epochs to wait before reducing learning rate
    /// </summary>
    public int Patience { get; init; }
    /// <summary>
    /// Tolerance to maintain when determining if learning progress has been made
    /// </summary>
    public float Tolerance { get; init; }
    /// <summary>
    /// Minimum learning grate
    /// </summary>
    public float MinimumLearningRate { get; init; }
    /// <summary>
    /// Underlying scheduler
    /// </summary>
    public ILearningRateScheduler Scheduler { get; init; }

    private float _bestLoss = float.MaxValue;
    private float _epochsSinceImprovement = 0;
    private float _currentMultiplier = 1;

    public ReduceLROnPlateau(ILearningRateScheduler scheduler, float reductionfactor = 0.5f, float recoveryfactor = 1.05f, int patience = 5, float tolerance = 1e-4f, float minRate = 1e-6f)
    {
        this.Scheduler = scheduler;
        this.ReductionFactor = reductionfactor;
        this.RecoveryFactor = recoveryfactor;
        this.Patience = patience;
        this.Tolerance = tolerance;
        this.MinimumLearningRate = minRate;
    }

    public void Step(int epoch, Metric<float> loss)
    {
        // Reset
        if (epoch == 0)
        {
            _bestLoss = float.MaxValue;
            _epochsSinceImprovement = 0;
            _currentMultiplier = 1;
        }
        // Do the step
        if (loss.Average < _bestLoss - Tolerance)
        {
            // Improvement: reset counter and gently recover LR
            _bestLoss = loss.Average;
            _epochsSinceImprovement = 0;
            _currentMultiplier = Math.Min(_currentMultiplier * RecoveryFactor, 1f); // cap at base rate
        }
        else
        {
            _epochsSinceImprovement++;
        }

        if (_epochsSinceImprovement >= Patience)
        {
            // Plateau reached: reduce LR
            _currentMultiplier = Math.Max(_currentMultiplier * ReductionFactor, MinimumLearningRate);
            _epochsSinceImprovement = 0;
        }

        Scheduler.Step(epoch, loss);
    }

    public float RateForEpoch(int epochFor, float baseRate)
    {
        float baseLr = Scheduler.RateForEpoch(epochFor, baseRate);
        return Math.Max(baseLr * _currentMultiplier, MinimumLearningRate);
    }
}