using DotML.Network.Training;

namespace DotML.Cli;

public interface ITrainingDataFormat {
    public bool IsInFormat(string path);
    public ITrainingDataSource<float> Read(string path);
}
