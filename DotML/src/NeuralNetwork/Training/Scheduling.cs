using System.Numerics;
using System.Reflection.Metadata;

namespace DotML.Network.Training;

/// <summary>
/// Scheduler for adjusting learning rate over epochs
/// </summary>
public interface ILearningRateScheduler
{
    /// <summary>
    /// Compute the learning rate for the given epoch from the starting/base rate
    /// </summary>
    /// <param name="baseRate">starting learning rate</param>
    /// <param name="epochFor">current epoch (o-index)</param>
    /// <returns>learning rate for this given epoch</returns>
    public float RateForEpoch(int epochFor);

    /// <summary>
    /// Step the scheduler at the end of the epoch
    /// </summary>
    /// <param name="epochFor">epoch that just ended</param>
    /// <param name="loss">loss computed for the epoch</param>
    public void Step(int epochFor, Metric<float> loss) {}
}

/// <summary>
/// Constant learning rate over all epochs
/// </summary>
public class ConstantRate : ILearningRateScheduler
{
    public float BaseRate {get; private set;}

    public ConstantRate(float baseRate)
    {
        this.BaseRate = baseRate;
    }

    public float RateForEpoch(int epochFor) => BaseRate;
}

/// <summary>
/// Exponential decay of learning rate over epochs
/// </summary>
public class ExponentialDecay : ILearningRateScheduler
{
    public float Decay { get; private set; }
    public int StepInterval { get; private set; }
    public float MinRate { get; private set; }
    public float BaseRate {get; private set;}

    public ExponentialDecay(float baseRate, float decay, int every = 1, float minRate = 1e-6f)
    {
        this.BaseRate = baseRate;
        this.Decay = Math.Abs(decay);
        this.StepInterval = Math.Max(1, every);
        this.MinRate = Math.Abs(minRate);
    }

    public static ExponentialDecay HalfEvery(float baseRate, int epochs) => new ExponentialDecay(baseRate, MathF.Log(2) / epochs);

    public float RateForEpoch(int epochFor)
    {
        var steps = epochFor / StepInterval;
        var rate = BaseRate * MathF.Exp(-Decay * steps);
        return MathF.Max(rate, this.MinRate);
    }
}

/// <summary>
/// Incrementally step down the learning each interval of epochs
/// </summary>
public class StepDecay : ILearningRateScheduler
{
    public float BaseRate {get; private set;}
    public float Gamma { get; private set; }
    public int StepInterval { get; private set; }

    public StepDecay(float baseRate, float gamma, int step)
    {
        this.BaseRate = baseRate;
        this.Gamma = Math.Abs(gamma);
        this.StepInterval = Math.Max(1, step);
    }

    public float RateForEpoch(int epochFor)
    {
        return BaseRate * MathF.Pow(Gamma, epochFor / StepInterval);
    }
}

/// <summary>
/// Reduce the learning rate using polynomial decay
/// </summary>
public class PolynomialDecay : ILearningRateScheduler
{
    public float BaseRate {get; private set;}
    public float Power { get; private set; }
    public int MaxEpochs { get; private set; }

    public PolynomialDecay(float baseRate, float power, int epochs)
    {
        this.BaseRate = baseRate;
        this.Power = Math.Abs(power);
        this.MaxEpochs = Math.Max(1, epochs);
    }

    public float RateForEpoch(int epochFor)
    {
        return BaseRate * MathF.Pow(1 - MathF.Min(epochFor, MaxEpochs) / MaxEpochs, Power);
    }
}

/// <summary>
/// Cosine annealing learning rate decay over a given number of epochs
/// </summary>
public class CosineAnnealing : ILearningRateScheduler
{
    public float BaseRate {get; private set;}
    public int MaxEpochs { get; private set; }

    public CosineAnnealing(float baseRate, int epochs)
    {
        this.BaseRate = baseRate;
        this.MaxEpochs = Math.Max(1, epochs);
    }

    public float RateForEpoch(int epochFor)
    {
        return BaseRate * 0.5f * (1 + MathF.Cos(MathF.PI * Math.Min(epochFor, MaxEpochs) / MaxEpochs));
    }
}

/// <summary>
/// A series of fixed rates that apply to different sections of epochs
/// </summary>
public class MilestoneRates : ILearningRateScheduler
{
    private (int Milestone, float Rate)[] steps;

    public MilestoneRates(params ReadOnlySpan<(int Milestone, float Rate)> steps)
    {
        this.steps = steps.ToArray();
        Array.Sort(this.steps, (a,b) => a.Milestone.CompareTo(b.Milestone));

        if (steps.Length == 0)
            throw new InvalidOperationException("MultiStepDecay must have at least one step.");
    }

    public float RateForEpoch(int epochFor)
    {
        epochFor = Math.Max(0, epochFor);

        // Find closest epoch
        int closest = 0; 
        for (var i = 0; i < steps.Length; i++)
        {
            if (steps[i].Milestone <= epochFor)
                closest = i;
            else 
                break;
        }
        return steps[closest].Rate;
    }
}

/// <summary>
/// A schedule that repeats after a given number of epochs
/// </summary>
public class CyclicSchedule : ILearningRateScheduler
{
    public int CycleLength {get; init;}
    public ILearningRateScheduler Scheduler {get; init;}

    public CyclicSchedule(int cycleLength, ILearningRateScheduler scheduler)
    {
        this.CycleLength = Math.Max(1, cycleLength);
        this.Scheduler = scheduler;
    }

    public float RateForEpoch(int epochFor)
    {
        // The epoch simply cycles in a loop of length cycleLength ie for length 4: [0, 1, 2, 3] [0, 1, 2, 3] ...
        var remainder = epochFor % CycleLength;
        return Scheduler.RateForEpoch(remainder);
    }

}

/// <summary>
/// Hold a constant value before starting scheduler
/// </summary>
public class ConstantRateWarmup : ILearningRateScheduler
{
    public float WarmupRate {get; set;}
    public int WarmupEpochs { get; init; }
    public ILearningRateScheduler Scheduler { get; init; }

    public ConstantRateWarmup(float warmupRate, int warmupEpochs, ILearningRateScheduler scheduler)
    {
        this.WarmupRate = warmupRate;
        this.WarmupEpochs = Math.Max(1, warmupEpochs);
        this.Scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
    }

    public float RateForEpoch(int epochFor)
    {
        if (epochFor < WarmupEpochs)
            return WarmupRate;                                    // We are in warmup, do nothing
        return Scheduler.RateForEpoch(epochFor - WarmupEpochs);   // Out of warmup, delegate to scheduler
    }

    public void Step(int epoch, Metric<float> loss) => Scheduler.Step(epoch, loss);
}

/// <summary>
/// Linearly increase from 0 to base-rate before starting the scheduler
/// </summary>
public class RampUpWarmup : ILearningRateScheduler
{
    public float MaxWarmupRate {get; set;}
    public int WarmupEpochs { get; init; }
    public ILearningRateScheduler Scheduler { get; init; }

    public RampUpWarmup(float maxWarmupRate, int warmupEpochs, ILearningRateScheduler scheduler)
    {
        this.MaxWarmupRate = maxWarmupRate;
        this.WarmupEpochs = Math.Max(1, warmupEpochs);
        this.Scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
    }

    public float RateForEpoch(int epochFor)
    {
        if (epochFor < WarmupEpochs)
            return MathF.Min(MaxWarmupRate, MaxWarmupRate * (epochFor + 1) / WarmupEpochs); // We are in warmup, ramp up
        return Scheduler.RateForEpoch(epochFor - WarmupEpochs);   // Out of warmup, delegate to scheduler
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

    public float RateForEpoch(int epochFor)
    {
        float baseLr = Scheduler.RateForEpoch(epochFor);
        return Math.Max(baseLr * _currentMultiplier, MinimumLearningRate);
    }
}