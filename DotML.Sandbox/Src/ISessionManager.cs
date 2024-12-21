using DotML.Network;

namespace DotML.Sandbox.Data;

public class Checkpoint {
    public string Name {get; init;}
    public DateTime Created {get; init;}
    public Checkpoint(string name, DateTime created) {
        this.Name = name;
        this.Created = created;
    }
}

public interface ICheckpointManager<T> {

    public void ClearCheckpoints();
    public IEnumerable<Checkpoint> ListCheckpoints();

    public Checkpoint CreateCheckpoint(T target);
    public void RestoreFromCheckpoint(T target, Checkpoint checkpoint);
}

/// <summary>
/// Checkpoint manager that never saves any checkpoint data, useful for dummy implementation 
/// </summary>
/// <typeparam name="T">type of thing to checkpoint</typeparam>
public class NullCheckpointManager<T> : ICheckpointManager<T> {
    public void ClearCheckpoints() {}

    public Checkpoint CreateCheckpoint(T target) {
        return new Checkpoint("null", DateTime.Now);
    }

    public IEnumerable<Checkpoint> ListCheckpoints() { yield break; }

    public void RestoreFromCheckpoint(T target, Checkpoint checkpoint) { }
}

/// <summary>
/// Basic checkpoint manager for ConvolutionalFeedforwardNetworks which only saves/loads from a single checkpoint
/// </summary>
public class CnnSingleCheckpointManager : ICheckpointManager<ConvolutionalFeedforwardNetwork> {

    private string session_key;
    private string filename;

    public CnnSingleCheckpointManager(string key) {
        this.session_key = key;
        filename = this.session_key + ".safetensors";
    }

    public void ClearCheckpoints() {
        if (File.Exists(filename))
            File.Delete(filename);
    }

    public Checkpoint CreateCheckpoint(ConvolutionalFeedforwardNetwork target) {
        var now = DateTime.Now;

        using var writer = new BinaryWriter(File.Open(filename, FileMode.Create));
        target.ToSafetensor(writer); 

        return new Checkpoint(filename, now);
    }

    public IEnumerable<Checkpoint> ListCheckpoints() {
        if (File.Exists(filename)) {
            yield return new Checkpoint(filename, DateTime.Now);
        }
    }

    public void RestoreFromCheckpoint(ConvolutionalFeedforwardNetwork target, Checkpoint checkpoint) {
        Safetensors sb = Safetensors.ReadFromFile(checkpoint.Name);
        target.FromSafetensor(sb);
    }
}