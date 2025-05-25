
using System.Text.Json;
using DotML.Network;

namespace DotML.Cli.Embeddings;

/// <summary>
/// Treat the input as a JSON vector (double array)
/// </summary>
public class Json : IEmbedder {

    public BatchedFeatureSet<float> CreateEmbedding(FeedforwardNetwork @for, IEnumerable<FileInfo> files) {
        var batches = files.Select(file => CreateEmbedding(@for, file)).ToArray();
        return new BatchedFeatureSet<float>(batches);
    }
    
    public FeatureSet<float> CreateEmbedding(FeedforwardNetwork @for, FileInfo file) {
        var vec = Matrix<float>.Column(JsonSerializer.Deserialize<float[]>(file.OpenRead()) ?? new float[0]);
        return new FeatureSet<float>(vec);
    }

    public BatchedFeatureSet<float> CreateEmbedding(FeedforwardNetwork @for, string raw) {
        var vec = Matrix<float>.Column(JsonSerializer.Deserialize<float[]>(raw) ?? new float[0]);
        return new BatchedFeatureSet<float>(new FeatureSet<float>(vec));
    }
}