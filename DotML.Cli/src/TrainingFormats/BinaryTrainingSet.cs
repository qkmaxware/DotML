using DotML.Network.Training;
using DotML.Network.Training.Formats;

namespace DotML.Cli.TrainingData;

/// <summary>
/// Treat the training data as a binary serialized TrainingSet object
/// </summary>
public class BinaryTrainingSet : ITrainingDataFormat {
    public bool IsInFormat(FileInfo file) {
        return TrainingSet<float>.IsBinaryTrainingSet(file);
    }
    public ITrainingDataSource<float> Read(FileInfo file) {
        BinaryTensorLoader<float> loader = new BinaryTensorLoader<float>();
        return loader.Load(file.FullName);
    }
}