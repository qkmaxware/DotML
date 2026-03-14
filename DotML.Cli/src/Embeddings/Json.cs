
using System.Text;
using System.Text.Json;
using DotML.Network;

namespace DotML.Cli.Embeddings;

/// <summary>
/// Treat the input as a JSON vector (double array)
/// </summary>
public class Json : IEmbedder {
    
    public Tensor<float> CreateEmbedding(INetworkModule @for, FileInfo file) {
        using var stream = file.OpenRead();
        return TensorExport.FromJson<float>(stream);
    }

    public Tensor<float> CreateEmbedding(INetworkModule @for, string raw) {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(raw ?? ""));
        return TensorExport.FromJson<float>(stream);
    }
}