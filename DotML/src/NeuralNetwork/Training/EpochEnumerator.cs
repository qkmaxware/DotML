namespace DotML.Network.Training;

/// <summary>
/// An enumerable that iterates during network training allowing for access to partial information in the middle of training
/// </summary>
/// <typeparam name="TNetwork"></typeparam>
public interface IEpochEnumerator<TNetwork> : IEnumerator<TNetwork> where TNetwork:INeuralNetwork {
    /// <summary>
    /// Current training epoch
    /// </summary>
    public int CurrentEpoch {get;}

    /// <summary>
    /// Advance the enumerator to the end of the sequence
    /// </summary>
    public void MoveToEnd() {
        while (this.MoveNext()) { }
    }
}