using System.Drawing;

namespace Qkmaxware.Terminal.Elements;

public abstract class Likelihood : IElement
{

    public char EmptyCharacter = DefaultEmptyCharacter;
    public char FilledCharacter = DefaultFilledCharacter;
    public char StartCharacter = DefaultStartCharacter;
    public char EndCharacter = DefaultEndCharacter;

    public Likelihood() { }

    protected abstract int GetClasses();
    protected abstract string GetLabelFor(int x);
    protected abstract float GetProbabilityFor(int x);

    const char DefaultEmptyCharacter = '-';
    const char DefaultFilledCharacter = '█';
    const char DefaultStartCharacter = '|';
    const char DefaultEndCharacter = '|';

    public LayoutSize Render(Graphics graphics)
    {
        // Example
        /*
        0:Cat      |-------         |130%
        1:Dog      |----------      |50%
        2:Elephant |-------         |15%
        3:Squirrel |--              |5%
        */
        var width = graphics.DrawingRegion.Width;

        var count = this.GetClasses();
        var max_label_length = 0;

        for (var i = 0; i < count; i++)
            max_label_length = Math.Max(max_label_length, GetLabelFor(i).Length);

        var remainingWidth = Math.Max(0, width - (max_label_length + 1));
        var barWidth = Math.Max(0, remainingWidth - 5);

        for (var i = 0; i < count; i++)
        {
            var label = GetLabelFor(i);
            var probability = Math.Clamp(GetProbabilityFor(i), 0f, 1f);
            graphics.DrawClipped(new Point(0, i), label.PadRight(max_label_length, ' '));

            graphics.Draw(new Point(max_label_length, i), StartCharacter);
            var divider = Math.Round(probability * barWidth);
            for (var j = 0; j < barWidth; j++)
            {
                graphics.Draw(new Point(max_label_length + 1 + j, i), j < divider ? FilledCharacter : EmptyCharacter);
            }
            graphics.Draw(new Point(max_label_length + 1 + barWidth, i), EndCharacter);
            graphics.DrawClipped(new Point(max_label_length + 2 + barWidth, i), $"{probability * 100:F0}%");
        }

        return new LayoutSize(width, count);
    }
}

public class FixedLikelihood : Likelihood
{
    private float[] prob;
    private string[]? labels;

    public FixedLikelihood(float[] probabilities, string[]? labels)
    {
        this.prob = probabilities;
        this.labels = labels;
    }

    protected override int GetClasses()
    {
        return prob.Length;
    }

    protected override string GetLabelFor(int x)
    {
        return labels is not null && x >= 0 && x < labels.Length ? labels[x] : $"Item {x + 1}";
    }

    protected override float GetProbabilityFor(int x)
    {
        return x >= 0 && x < prob.Length ? prob[x] : 0.0f;
    }
}

public class DynamicLikelihood : Likelihood
{
    private int classes;
    public DynamicLikelihood(int classes)
    {
        this.classes = classes;
    }

    public DynamicLikelihood(int classes, Func<int, float> src, Func<int, string> labels)
    {
        this.classes = classes;
        this.Source = src;
        this.Labeller = labels;
    }

    public Func<int, string>? Labeller { get; init; }
    public Func<int, float>? Source { get; init; }

    protected override int GetClasses()
    {
        return classes;
    }

    protected override string GetLabelFor(int x)
    {
        return Labeller?.Invoke(x) ?? $"Item {x}"; 
    }

    protected override float GetProbabilityFor(int x)
    {
        return Source?.Invoke(x) ?? 0f;
    }
}