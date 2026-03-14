using System.Diagnostics.CodeAnalysis;
using System.Security.Cryptography;

namespace DotML.Federated;

public abstract class Worker<TModel, TWeights>
{
    public Guid WorkerId { get; } = Guid.NewGuid();

    protected int Seed {get; private set;}
    protected Random Random {get; private set;}

    public Worker()
    {
        Seed = RandomNumberGenerator.GetInt32(int.MinValue, int.MaxValue);
        Random = new Random(Seed);
    }
}

public abstract class TrainerWorker<TModel, TWeights> : Worker<TModel, TWeights>
{
    private ModelVersion clientModelVersion;
    private TModel clientModel;

    public TrainerWorker(TModel initial) : base()
    {
        this.clientModel = initial;
        this.clientModelVersion = new ModelVersion(0);    
    }

    private void TrainingIteration()
    {
        // 1. Pull global model
        if (!TryPullGlobalWeights(out var globalVersion, out var globalWeighs)) {
            return;
        }

        // Update local model and version
        this.clientModelVersion = globalVersion;
        this.UpdateLocalModel(ref this.clientModel, globalWeighs);

        // 2. Train local model
        Train(ref this.clientModel, out var samplesTrained);

        // 3. Push updated weights
        var updatedWeights = ExtractWeights(this.clientModel);
        PushLocalWeights(this.clientModelVersion, samplesTrained, updatedWeights);
    }

    public void TrainSync(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            TrainingIteration();
        }
    }

    public Task TrainAsync(CancellationToken token)
    {
        return Task.Run(() =>
        {
            while (!token.IsCancellationRequested)
            {
                TrainingIteration();
            }
        }, token);
    }

    protected abstract bool TryPullGlobalWeights(out ModelVersion version, [NotNullWhen(true)]out TWeights? weights);
    protected abstract void UpdateLocalModel(ref TModel model, TWeights weights);
    protected abstract void Train(ref TModel model, out int samplesCompleted);
    protected abstract TWeights ExtractWeights(TModel model);
    protected abstract void PushLocalWeights(ModelVersion version, int samplesTrained, TWeights weights);
}