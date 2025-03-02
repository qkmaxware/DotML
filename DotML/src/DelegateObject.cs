using System.Runtime.CompilerServices;

namespace DotML;

/// <summary>
/// An object that behaves like a delegate / function pointer
/// </summary>
public abstract class DelegateObject<TIn, TOut> {

    // Since we can't overload () overload [] instead

    /// <summary>
    /// Invoke the delegate function with the given input
    /// </summary>
    /// <param name="input">input value</param>
    /// <returns>function output</returns>
    public TOut this[TIn input] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this.Invoke(input);
    }

    /// <summary>
    /// Invoke the function with the given input
    /// </summary>
    /// <param name="input">input value</param>
    /// <returns>function output</returns>
    public abstract TOut Invoke(TIn input);
}

/// <summary>
/// An object that behaves like a delegate / function pointer
/// </summary>
public abstract class DelegateObject<TIn1, TIn2, TOut> {

    // Since we can't overload () overload [] instead

    /// <summary>
    /// Invoke the delegate function with the given input
    /// </summary>
    /// <param name="input1">first input value</param>
    /// <param name="input2">second input value</param>
    /// <returns>function output</returns>
    public TOut this[TIn1 input1, TIn2 input2] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this.Invoke(input1, input2);
    }

    /// <summary>
    /// Invoke the function with the given inputs
    /// </summary>
    /// <param name="input1">first input value</param>
    /// <param name="input2">second input value</param>
    /// <returns>function output</returns>
    public abstract TOut Invoke(TIn1 input1, TIn2 input2);
}