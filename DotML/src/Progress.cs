namespace DotML;

/// <summary>
/// A progress track that uses a series of "steps"
/// </summary>
public interface ISteppedProgress: IProgress<float>
{
    /// <summary>
    /// Index of the current step
    /// </summary>
    public int CurrentStep {get;}
    // Total number of steps
    public int StepCount {get;}

    /// <summary>
    /// Number of steps to advance by
    /// </summary>
    /// <param name="steps">step count</param>
    public void Advance(int steps = 1);
}

/// <summary>
/// Base implementation for a stepped progress tracker
/// </summary>
public class SteppedProgress: ISteppedProgress
{
    private Action<float> onStep;
    public int CurrentStep {get; private set;}
    public int StepCount {get; init;}
    public float CompletionPercent {get; private set;}

    public SteppedProgress(int steps): this(steps, static (progress) => { /* do nothing */ }) {}

    public SteppedProgress(int steps, Action<float> onStep)
    {
        this.StepCount = Math.Max(1, steps);
        this.onStep = onStep;
        Reset();
    }

    public void Reset()
    {
        this.CurrentStep = 0;
        this.CompletionPercent = 0.0f;
    }

    public void Report(float progress)
    {
        this.CompletionPercent = progress;
        this.onStep?.Invoke(progress);
    }

    public void Advance(int steps = 1)
    {
        var nextStep = Math.Clamp(CurrentStep + steps, 0, StepCount - 1);
        var percent = (float)nextStep / (float)this.StepCount;
        this.CurrentStep = nextStep;
        Report(percent);
    }
}