namespace DotML.Federated;

/// <summary>
/// Coordinator for federated learning.
/// </summary>
/// <typeparam name="TModel">Model type</typeparam>
/// <typeparam name="TWeights">Weight type</typeparam>
public abstract class Coordinator<TModel, TWeights>
{
    
    private readonly ReaderWriterLockSlim _lock = new();

    private ModelVersion globalModelVersion;
    private TModel globalModel;

    public Coordinator(TModel initialModel)
    {
        globalModel = initialModel;
        globalModelVersion = new ModelVersion(0);
    }

    /// <summary>
    /// Fetches the current global model weights and version
    /// </summary>
    /// <param name="version">model version</param>
    /// <returns>fetched weights</returns>
    public TWeights FetchWeights(out ModelVersion version)
    {
        _lock.EnterReadLock();
        try
        {
            // Assuming TModel has a method to extract weights
            return ExtractWeights(globalModel, out version);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    } 

    /// <summary>
    /// Extracts weights from the model
    /// </summary>
    /// <param name="model">model instance</param>
    /// <param name="version">model version</param>
    /// <returns>weights</returns>
    protected abstract TWeights ExtractWeights(TModel model, out ModelVersion version);

    /// <summary>
    /// Pushes updated weights from a client to the coordinator
    /// </summary>
    /// <param name="clientVersion">model version on the client worker</param>
    /// <param name="samplesTrained">number of samples trained on the client worker</param>
    /// <param name="weights">the updated weights</param>
    public void PushWeights(ModelVersion clientVersion, int samplesTrained, TWeights weights)
    {
        _lock.EnterWriteLock();
        try
        {
            UpdateGlobalModel(ref this.globalModel, ref this.globalModelVersion, clientVersion, samplesTrained, weights);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Updates the global model with the client's update
    /// </summary>
    /// <param name="model">model</param>
    /// <param name="globalVersion">global version</param>
    /// <param name="clientVersion">client worker model version</param>
    /// <param name="clientSamples">number of samples trained on the client worker</param>
    /// <param name="update">updated weights</param>
    protected abstract void UpdateGlobalModel(ref TModel model, ref ModelVersion globalVersion, ModelVersion clientVersion, int clientSamples, TWeights update);
}
