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
    public float RateForEpoch(float baseRate, int epochFor);
}

public class ConstantRate : ILearningRateScheduler
{
    public float RateForEpoch(float baseRate, int epochFor) => baseRate;
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

    public float RateForEpoch(float baseRate, int epochFor)
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

    public float RateForEpoch(float baseRate, int epochFor)
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

    public float RateForEpoch(float baseRate, int epochFor)
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

    public float RateForEpoch(float baseRate, int epochFor)
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

    public float RateForEpoch(float baseRate, int epochFor)
    {
        if (epochFor < WarmupEpochs)
            return baseRate;                                                // We are in warmup, do nothing
        return Scheduler.RateForEpoch(baseRate, epochFor - WarmupEpochs);   // Out of warmup, delegate to scheduler
    }
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

    public float RateForEpoch(float baseRate, int epochFor)
    {
        if (epochFor < WarmupEpochs)
            return MathF.Min(baseRate, baseRate * (epochFor + 1) / WarmupEpochs); // We are in warmup, ramp up
        return Scheduler.RateForEpoch(baseRate, epochFor - WarmupEpochs);   // Out of warmup, delegate to scheduler
    }
}