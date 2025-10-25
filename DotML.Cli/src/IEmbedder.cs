using DotML.Network;

namespace DotML.Cli;

public interface IEmbedder {
    public Tensor<float> CreateEmbedding(INetworkModule @for, FileInfo file);
    public Tensor<float> CreateEmbedding(INetworkModule @for, string raw);
}