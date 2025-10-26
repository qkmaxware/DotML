using DotML.Network.Training;

namespace DotML.Cli;

public abstract class DirectoryTrainingDataFormat : ITrainingDataFormat
{
    public bool IsInFormat(string path)
    {
        if (Directory.Exists(path))
            return IsInFormat(new DirectoryInfo(path));

        return false;
    }

    public abstract bool IsInFormat(DirectoryInfo file);

    public ITrainingDataSource<float> Read(string path)
    {
        if (Directory.Exists(path))
            return Read(new DirectoryInfo(path));

        throw new NotImplementedException();
    }

    public abstract ITrainingDataSource<float> Read(DirectoryInfo file);
}