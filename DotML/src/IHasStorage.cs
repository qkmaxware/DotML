namespace DotML;

/// <summary>
/// Interface to indicate that this object have it's storage size computed
/// </summary>
public interface IHasStorage {
    /// <summary>
    /// Amount of storage consumed by this object
    /// </summary>
    /// <returns>data size</returns>
    public DataSize StorageSize();
}