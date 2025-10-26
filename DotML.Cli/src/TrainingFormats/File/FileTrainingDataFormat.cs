using DotML.Network.Training;

namespace DotML.Cli;

public abstract class FileTrainingDataFormat : ITrainingDataFormat
{
    public bool IsInFormat(string path)
    {
        if (File.Exists(path))
            return IsInFormat(new FileInfo(path));
        
        return false;
    }

    public abstract bool IsInFormat(FileInfo file);
    
    public ITrainingDataSource<float> Read(string path)
    {
        if (File.Exists(path))
            return Read(new FileInfo(path));

        throw new NotImplementedException();
    }

    public abstract ITrainingDataSource<float> Read(FileInfo file); 
}