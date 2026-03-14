using System.Diagnostics.CodeAnalysis;

namespace DotML.Federated;

public readonly struct ModelVersion: IParsable<ModelVersion>
{
    public long GlobalStep {get;}

    public ModelVersion(long globalStep)
    {
        GlobalStep = globalStep;
    }

    public static ModelVersion Parse(string? s, IFormatProvider? provider)
    {
        return new ModelVersion(long.Parse(s ?? string.Empty, provider));
    }

    public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, [MaybeNullWhen(false)] out ModelVersion result)
    {
        if (long.TryParse(s, System.Globalization.NumberStyles.Integer, provider, out var step))
        {
            result = new ModelVersion(step);
            return true;
        }
        result = default;
        return false;
    }

    public static bool operator > (ModelVersion left, ModelVersion right)
    {
        return left.GlobalStep > right.GlobalStep;
    }

    public static bool operator < (ModelVersion left, ModelVersion right)
    {
        return left.GlobalStep < right.GlobalStep;
    }

    public static bool operator == (ModelVersion left, ModelVersion right)
    {
        return left.GlobalStep == right.GlobalStep;
    }

    public static bool operator != (ModelVersion left, ModelVersion right)
    {
        return left.GlobalStep != right.GlobalStep;
    }

    public override bool Equals(object? obj)
    {
        return obj is ModelVersion version && this == version;
    }

    public override int GetHashCode()
    {
        return GlobalStep.GetHashCode();
    }
}