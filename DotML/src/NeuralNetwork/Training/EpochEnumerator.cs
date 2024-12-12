namespace DotML.Network.Training;

public delegate void BatchStartHandler(int batch, int batchCount);
public delegate void BatchEndHandler(int batch, int batchCount);
public delegate void EpochStartHandler(int epoch, int epochCount);
public delegate void ValidationStartHandler(int epoch, int epochCount);
public delegate void ValidationStepHandler(int epoch, int epochCount, int inputIndex, double accuracy);
public delegate void ValidationEndHandler(int epoch, int epochCount, double accuracy);
public delegate void EpochEndHandler(int epoch, int epochCount);

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
    /// Max number of epochs
    /// </summary>
    public int MaxEpochs {get;}

    /// <summary>
    /// Advance the enumerator to the end of the sequence
    /// </summary>
    public void MoveToEnd() {
        while (this.MoveNext()) { }
    }

    public event EpochStartHandler OnEpochStart;
    public event BatchStartHandler OnBatchStart;
    public event BatchEndHandler OnBatchEnd;
    public event ValidationStartHandler OnValidationStart;
    public event ValidationStepHandler OnValidated;
    public event ValidationEndHandler OnValidationEnd;
    public event EpochEndHandler OnEpochEnd;
}