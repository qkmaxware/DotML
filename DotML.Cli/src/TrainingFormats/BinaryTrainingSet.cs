using DotML.Network.Training;

namespace DotML.Cli.TrainingData;

/// <summary>
/// Treat the training data as a binary serialized TrainingSet object
/// </summary>
public class BinaryTrainingSet : ITrainingDataFormat {
    public bool IsInFormat(FileInfo file) {
        return TrainingSet.IsBinaryTrainingSet(file);
    }
    public TrainingSet Read(FileInfo file) {
        TrainingSet set = new TrainingSet();

        using var reader = new BinaryReader(file.OpenRead());
        set.AddFrom(reader);

        return set;
    }
}