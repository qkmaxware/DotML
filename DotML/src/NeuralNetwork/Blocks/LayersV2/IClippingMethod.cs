using System.Numerics;
using System.Runtime.CompilerServices;

namespace DotML.Network;

/// <summary>
/// Clipping strategy for gradients that are applied to specific gradient tensors individually
/// </summary>
public interface ILocalClippingStrategy<TNum>
where TNum : INumber<TNum>
{
    /// <summary>
    /// Clip an input gradient
    /// </summary>
    /// <param name="grad">gradient tensor</param>
    public void ClipInput(Tensor<TNum> grad);
    /// <summary>
    /// Clip an weight gradient
    /// </summary>
    /// <param name="grad">gradient tensor</param>
    public void ClipWeight(Tensor<TNum> grad);
    /// <summary>
    /// Clip an bias gradient
    /// </summary>
    /// <param name="grad">gradient tensor</param>
    public void ClipBias(Tensor<TNum> grad);
}

/// <summary>
/// Clipping strategy where gradients are clipped via their absolute magnitude symetrically about 0. NaN sanitization is also performed.
/// </summary>
public class LocalMagnitudeClipping<TNum> 
: ILocalClippingStrategy<TNum>
where TNum:INumber<TNum>
{
    public TNum? InputMagnitude { get; init; }
    public TNum? WeightMagnitude { get; init; }
    public TNum? BiasMagnitude { get; init; }

    /// <summary>
    /// Clip all gradients using the same gradient
    /// </summary>
    /// <param name="magnitude">gradient max magnitude</param>
    public LocalMagnitudeClipping(TNum magnitude) : this(magnitude, magnitude, magnitude) { }

    /// <summary>
    /// Clip all gradients using different values for each kind of gradient, ignore input gradient clipping
    /// </summary>
    /// <param name="weight">max magnitude for weight gradients</param>
    /// <param name="bias">max magnitude for bias gradients</param>
    public LocalMagnitudeClipping(TNum? weight, TNum? bias)
    {
        this.InputMagnitude = default;
        this.WeightMagnitude = weight != null ? TNum.Abs(weight) : default;
        this.BiasMagnitude = bias != null ? TNum.Abs(bias) : default;
    }

    /// <summary>
    /// Clip all gradients using different values for each kind of gradient
    /// </summary>
    /// <param name="input">max magnitude for input gradients</param>
    /// <param name="weight">max magnitude for weight gradients</param>
    /// <param name="bias">max magnitude for bias gradients</param>
    public LocalMagnitudeClipping(TNum? input, TNum? weight, TNum? bias)
    {
        this.InputMagnitude = input != null ? TNum.Abs(input) : default;
        this.WeightMagnitude = weight != null ? TNum.Abs(weight) : default;
        this.BiasMagnitude = bias != null ? TNum.Abs(bias) : default;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private TNum Clip(TNum x, TNum magnitude)
    {
        return TNum.Abs(x) > magnitude ? TNum.CopySign(magnitude, sign: x) : x;
    }

    public void ClipInput(Tensor<TNum> grad)
    {
        if (InputMagnitude != null)
            grad.ElementWiseInplace((x) => Clip(x, InputMagnitude));
    }

    public void ClipWeight(Tensor<TNum> grad)
    {
        if (WeightMagnitude != null)
            grad.ElementWiseInplace((x) => Clip(x, WeightMagnitude));
    }

    public void ClipBias(Tensor<TNum> grad)
    {
        if (BiasMagnitude != null)
            grad.ElementWiseInplace((x) => Clip(x, BiasMagnitude));
    }

}

/// <summary>
/// Clipping strategy which is applied across all gradients at the same time
/// </summary>
public interface IGlobalClippingStrategy<TNum>
where TNum : INumber<TNum>
{
    /// <summary>
    /// Clip all parameters provided, clipping is done in-place editing the provided tensors
    /// </summary>
    /// <param name="parameters">enumerable of tensors to clip</param>
    public void Clip(IEnumerable<Tensor<TNum>> parameters);
}

/// <summary>
/// Clipping strategy where gradients are clipped via their absolute magnitude symetrically about 0.
/// </summary>
/// <typeparam name="TNum"></typeparam>
public class GlobalMagnitudeClipping<TNum>
: IGlobalClippingStrategy<TNum>
where TNum : INumber<TNum>
{
    /// <summary>
    /// Magnitude to clip gradients larger than
    /// </summary>
    public TNum Magnitude { get; init; }

    /// <summary>
    /// Create a new clipping strategy with the given maximum magnitude
    /// </summary>
    /// <param name="magnitude">maximum gradient magnitude</param>
    public GlobalMagnitudeClipping(TNum magnitude)
    {
        this.Magnitude = TNum.Abs(magnitude);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private TNum Clip(TNum x)
    {
        return TNum.Abs(x) > Magnitude ? TNum.CopySign(Magnitude, sign: x) : x;
    }

    public void Clip(IEnumerable<Tensor<TNum>> parameters)
    {
        foreach (var tensor in parameters)
            tensor.ElementWiseInplace((x) => Clip(x));
    }

}

/// <summary>
/// Implementation of global norm clipping 
/// </summary>
/// <typeparam name="TNum">numeric type to clip</typeparam>
public class GlobalNormClipping<TNum>
: IGlobalClippingStrategy<TNum>
where TNum : INumber<TNum>, IRootFunctions<TNum>
{
    /// <summary>
    /// Maximum allowed global L2 norm (magnitude/length)of all parameter gradients. If the total norm exceeds this value, the gradients are scaled down proportionally
    /// </summary>
    public TNum MaxNorm { get; init; }

    /// <summary>
    /// Create a new global norm clipping strategy with a maximum norm of 1.0
    /// </summary>
    public GlobalNormClipping() : this(TNum.One) { }

    /// <summary>
    /// Create a new global norm clipping strategy with a provided maximum norm
    /// </summary>
    /// <param name="max">maximum L2 norm value</param>
    public GlobalNormClipping(TNum max)
    {
        this.MaxNorm = TNum.Abs(max);
    }

    public void Clip(IEnumerable<Tensor<TNum>> parameters)
    {
        TNum globalNormSquared = TNum.Zero;

        // Compute global norm squared
        foreach (var tensor in parameters)
        {
            foreach (var value in tensor.AsSpan())
            {
                globalNormSquared += value * value;
            }
        }

        TNum globalNorm = TNum.Sqrt(globalNormSquared);

        if (globalNorm > MaxNorm && globalNorm > TNum.Zero)
        {
            TNum scale = MaxNorm / globalNorm;

            // Scale all tensors
            foreach (var tensor in parameters)
            {
                tensor.ScaleByInplace(scale);
            }
        }
    }

}