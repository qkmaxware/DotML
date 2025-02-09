
using System.Text.Json;
using DotML.Network;

namespace DotML.Cli.Embeddings;

/// <summary>
/// Treat the input as a JSON vector (double array)
/// </summary>
public class Json : IEmbedder {
    public BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, FileInfo file) {
        var vec = Matrix<double>.Column(JsonSerializer.Deserialize<double[]>(file.OpenRead()) ?? new double[0]);
        return new BatchedFeatureSet<double>(new FeatureSet<double>(vec));
    }

    public BatchedFeatureSet<double> CreateEmbedding(FeedforwardNetwork @for, string raw) {
        var vec = Matrix<double>.Column(JsonSerializer.Deserialize<double[]>(raw) ?? new double[0]);
        return new BatchedFeatureSet<double>(new FeatureSet<double>(vec));
    }
}