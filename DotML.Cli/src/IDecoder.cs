namespace DotML.Cli;

public interface IDecoder {
    public IDecodedResult Decode(BatchedFeatureSet<double> output_values);
}

public interface IDecodedResult : IDisposable {
    public void ConsoleOutput();
    public void FileOutput(FileInfo file);
}