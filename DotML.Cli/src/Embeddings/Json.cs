
using System.Text.Json;

namespace DotML.Cli.Embeddings;

/// <summary>
/// Treat the input as a JSON vector (double array)
/// </summary>
public class Json : IEmbedder {
    public Vec<double> CreateEmbedding(FileInfo file) {
        var vec = JsonSerializer.Deserialize<double[]>(file.OpenRead()) ?? new double[0];
        return Vec<double>.Wrap(vec);
    }

    public Vec<double> CreateEmbedding(string raw) {
        var vec = JsonSerializer.Deserialize<double[]>(raw) ?? new double[0];
        return Vec<double>.Wrap(vec);
    }
}