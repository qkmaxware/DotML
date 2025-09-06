using System.Text.Json;
using DotML.Network.Training;

namespace DotML.Cli.TrainingData;

/// <summary>
/// Treat the training data as a JSON array of Input/Output vector pairs
/// </summary>
public class JsonVectorPairs : ITrainingDataFormat {

    private struct IOPair {
        public float[]? Input {get; set;}
        public float[]? Output {get; set;}
    }

    public bool IsInFormat(FileInfo file) {
        return file.Extension == ".json";
    }

    public TrainingSet<float> Read(FileInfo file) {
        #nullable disable //I am checking null on io.Input and io.Output but the static analyzer can't determine that from the linq flow
        return new TrainingSet<float>((JsonSerializer.Deserialize<IOPair[]>(file.OpenRead()) ?? new IOPair[0]).Where(io => io.Input is not null && io.Output is not null).Select(io => new TrainingPair<float>{ Input = Vec<float>.Wrap(io.Input), Output = Vec<float>.Wrap(io.Output)}));
        #nullable restore
    }
}