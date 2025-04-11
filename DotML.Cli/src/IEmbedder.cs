using DotML.Network;

namespace DotML.Cli;

public interface IEmbedder {
    public BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, IEnumerable<FileInfo> files);
    public BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, string raw);
}