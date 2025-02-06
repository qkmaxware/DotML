namespace DotML.Cli;

public interface IDecoder {
    public IDecodedResult Decode(Shape3D output_shape, Vec<double> output_values);
}

public interface IDecodedResult : IDisposable {
    public void ConsoleOutput();
    public void FileOutput(FileInfo file);
}