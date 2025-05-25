using DotML.Network.Training;

namespace DotML.Cli;

public interface ITrainingDataFormat {
    public bool IsInFormat(FileInfo file);
    public TrainingSet<float> Read(FileInfo file);
}
