namespace DotML.Cli;

public interface IDecoder {
    public IDecodedResult Decode(Vec<double> output);
}

public interface IDecodedResult {
    public void ConsoleOutput();
    public void FileOutput(FileInfo file);
}