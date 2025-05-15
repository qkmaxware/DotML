namespace DotML.Cli;

public interface IDecoder {
    public IDecodedResult Decode(BatchedFeatureSet<double> output_values);
}

public interface IFileOnlyDecoder : IDecoder {
    public bool FileRequired() => false;
}

public interface IDecodedResult : IDisposable {
    public void ConsoleOutput();
    public IEnumerable<FileInfo> FileOutput(FileInfo file);
}