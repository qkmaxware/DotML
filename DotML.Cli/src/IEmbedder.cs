using DotML.Network;

namespace DotML.Cli;

public interface IEmbedder {
    public BatchedFeatureSet<float> CreateEmbedding(FeedforwardNetwork @for, IEnumerable<FileInfo> files);
    public BatchedFeatureSet<float> CreateEmbedding(FeedforwardNetwork @for, string raw);
}