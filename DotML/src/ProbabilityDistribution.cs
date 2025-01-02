using System.Text;

namespace DotML;

/// <summary>
/// Probability distribution for a series of categories. Useful when used with output vectors from neural networks.
/// </summary>
public struct ProbabilityDistribution
{
    private double[] values;
    private string[]? labels;

    public int Categories => values.Length;

    public ProbabilityDistribution(Vec<double> vec)
    {
        this.values = (double[])vec.SoftmaxNormalized();
    }

    public ProbabilityDistribution(Vec<double> vec, string[]? labels)
    {
        this.values = (double[])vec.SoftmaxNormalized();
        this.labels = labels;
    }

    /// <summary>
    /// Maximum recorded probability
    /// </summary>
    public double MaxProbability => values is null || values.Length < 1 ? 0 : values.Max();

    /// <summary>
    /// Minimum recorded probability
    /// </summary>
    public double MinProbability => values is null || values.Length < 1 ? 0 : values.Min();

    /// <summary>
    /// Get the label for a particular category
    /// </summary>
    /// <returns>category label or nothing if there is no label</returns>
    public string? GetCategoryLabel(int index)
    {
        if (labels is null)
            return null;
        if (index >= 0 && index < labels.Length)
            return labels[index];
        else
            return null;
    }

    /// <summary>
    /// Get the probability of the given option
    /// </summary>
    /// <param name="index">option index</param>
    /// <returns>probability</returns>
    public double GetProbabilityOf(int index) {
        if (index < 0 || index >= values.Length)
            return 0;
        return 100 * values[index];
    }

    /// <summary>
    /// Select a category by choosing the one with the highest probability.
    /// </summary>
    /// <returns>index of the selected category</returns>
    public int SelectMostProbable()
    {
        int index = 0; double max = 0;
        for (int i = 0; i < this.Categories; i++)
        {
            if (i == 0 || this.values[i] > max)
            {
                index = i;
                max = this.values[i];
            }
        }

        return index;
    }

    /// <summary>
    /// Select a category by choosing the one with the highest probability.
    /// </summary>
    /// <returns>index of the selected category</returns>
    public int SelectMostProbable(out string? label)
    {
        int index = SelectMostProbable();
        label = GetCategoryLabel(index);
        return index;
    }

    private static readonly Random random = new Random();

    /// <summary>
    /// Select a category randomly taking into account the probability of selection for each category.
    /// </summary>
    /// <returns>index of the selected category</returns>
    public int SelectRandomly()
    {
        double selection = random.NextDouble();
        double cumulativeSum = 0d;
        for (var i = 0; i < this.Categories; i++)
        {
            cumulativeSum += this.values[i];
            if (selection < cumulativeSum)
            {
                return i;
            }
        }

        // Return last element if all else fails
        return this.Categories - 1;
    }

    /// <summary>
    /// Select a category randomly taking into account the probability of selection for each category.
    /// </summary>
    /// <returns>index of the selected category</returns>
    public int SelectRandomly(out string? label)
    {
        var index = SelectRandomly();
        label = GetCategoryLabel(index);
        return index;
    }

    public IEnumerable<(int Index, string? Label, double Probability)> EnumerateProbabilities() {
        for (var i = 0; i < this.values.Length; i++) {
            yield return (i, this.GetCategoryLabel(i), this.values[i]);
        }
    }

    public override string ToString()
    {
        // Example of distribution being printed
        /*
        0:Cat      ######## 30%
        1:Dog      ########### 50%
        2:Elephant ##### 15%
        3:Squirrel ## 5%
        */
        StringBuilder sb = new StringBuilder();
        var index = 0;
        var label_width = 0;
        if (this.labels is not null)
        {
            foreach (var label in labels)
            {
                if (label.Length > label_width)
                    label_width = label.Length;
            }
        }

        const int max_width = 30;
        foreach (var prob in this.values)
        {
            var label = this.GetCategoryLabel(index) ?? string.Empty;
            sb.Append(index); sb.Append(":"); sb.Append(label.PadRight(label_width, ' ')); sb.Append(' ');
            sb.Append(new String('#', (int)(max_width * prob))); sb.Append(' '); sb.Append((prob * 100).ToString("F2")); sb.AppendLine("%");
            index++;
        }

        return sb.ToString();
    }

}