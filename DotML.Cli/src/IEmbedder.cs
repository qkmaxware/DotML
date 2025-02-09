using DotML.Network;

namespace DotML.Cli;

public interface IEmbedder {
    public BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, FileInfo file);
    public BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, string raw);
}