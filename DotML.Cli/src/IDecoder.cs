using Qkmaxware.Terminal;

namespace DotML.Cli;

public interface IDecoder {
    public IDecodedResult Decode(Tensor<float> output_values);
}

public interface IFileOnlyDecoder : IDecoder {
    public bool FileRequired() => false;
}

public interface IDecodedResult : IDisposable {
    public IElement ConsoleOutput();
    public IEnumerable<FileInfo> FileOutput(FileInfo file);
}