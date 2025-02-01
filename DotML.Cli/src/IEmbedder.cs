namespace DotML.Cli;

public interface IEmbedder {
    public Vec<double> CreateEmbedding(FileInfo file);
    public Vec<double> CreateEmbedding(string raw);
}