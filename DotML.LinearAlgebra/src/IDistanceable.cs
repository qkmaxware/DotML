namespace DotML;

/// <summary>
/// An object that can have it's distance to another object measured
/// </summary>
/// <typeparam name="TInst">Object it can have it's distance measured to</typeparam>
/// <typeparam name="RScalar">Scalar type denoting the distance</typeparam>
public interface IDistanceable<TInst, RScalar> {
    /// <summary>
    /// Compute the distance from the current object to the given one
    /// </summary>
    /// <param name="instance">Object to have distance measured to</param>
    /// <returns>distance</returns>
    public RScalar DistanceTo(TInst instance);
}