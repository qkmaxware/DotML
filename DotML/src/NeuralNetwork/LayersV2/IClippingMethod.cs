using System.Runtime.CompilerServices;

namespace DotML.Network;

/// <summary>
/// Clipping strategy for gradients
/// </summary>
public interface IClippingStrategy {
    /// <summary>
    /// Clip an input gradient
    /// </summary>
    /// <param name="x">gradient value</param>
    /// <returns>clipped value</returns>
    public float ClipInput(float x);
    /// <summary>
    /// Clip an weight gradient
    /// </summary>
    /// <param name="x">gradient value</param>
    /// <returns>clipped value</returns>
    public float ClipWeight(float x);
    /// <summary>
    /// Clip an bias gradient
    /// </summary>
    /// <param name="x">gradient value</param>
    /// <returns>clipped value</returns>
    public float ClipBias(float x);
}

/// <summary>
/// Clipping strategy where gradients are clipped via their absolute magnitude symetrically about 0. NaN sanitization is also performed.
/// </summary>
public class MagnitudeClipping: IClippingStrategy {
    public float InputMagnitude {get; init;}
    public float WeightMagnitude {get; init;}
    public float BiasMagnitude {get; init;}

    /// <summary>
    /// Clip all gradients using the same gradient
    /// </summary>
    /// <param name="magnitude">gradient max magnitude</param>
    public MagnitudeClipping(float magnitude): this(magnitude, magnitude, magnitude) { }

    /// <summary>
    /// Clip all gradients using different values for each kind of gradient
    /// </summary>
    /// <param name="input">max magnitude for input gradients</param>
    /// <param name="weight">max magnitude for weight gradients</param>
    /// <param name="bias">max magnitude for bias gradients</param>
    public MagnitudeClipping(float input, float weight, float bias) {
        this.InputMagnitude = MathF.Abs(input);
        this.WeightMagnitude = MathF.Abs(weight);
        this.BiasMagnitude = MathF.Abs(bias);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float SanitizeNaN(float x) {
        return float.IsNaN(x) ? 1e-8f : x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float Clip(float x, float magnitude) {
        return Math.Abs(x) > magnitude ? Math.Sign(x) * magnitude : x;
    }

    public float ClipInput(float x) => Clip(SanitizeNaN(x), this.InputMagnitude);
    public float ClipWeight(float x) => Clip(SanitizeNaN(x), this.WeightMagnitude);
    public float ClipBias(float x) => Clip(SanitizeNaN(x), this.BiasMagnitude);
}