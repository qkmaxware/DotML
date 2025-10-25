using System.Text.Json;
using DotML.Network.Training;

namespace DotML.Cli.TrainingData;

/// <summary>
/// Treat the training data as a JSON array of Input/Output vector pairs.
/// Json data should look like [{"input": tensor, "output": tensor}]
/// </summary>
public class JsonVectorPairs : ITrainingDataFormat {

    public bool IsInFormat(FileInfo file) {
        return file.Extension == ".json";
    }

    public ITrainingDataSource<float> Read(FileInfo file) {
        using var stream = file.OpenRead();
        var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        List<(Tensor<float>, Tensor<float>)> ioPairs = new List<(Tensor<float>, Tensor<float>)>();
        TensorShape iShape = new TensorShape();
        TensorShape oShape = new TensorShape();

        if (root.ValueKind == JsonValueKind.Array)
        {
            var enumerator = root.EnumerateArray();
            while (enumerator.MoveNext())
            {
                var pair = enumerator.Current;
                if (pair.ValueKind != JsonValueKind.Object)
                    continue;

                JsonElement input;
                var has_input = pair.TryGetProperty("Input", out input) || pair.TryGetProperty("input", out input);

                JsonElement output;
                var has_output = pair.TryGetProperty("Output", out output) || pair.TryGetProperty("output", out output);

                if (!has_input || !has_output)
                    continue;

                Tensor<float> iTensor = TensorExport.FromJson<float>(input);
                iShape = iTensor.Shape;
                Tensor<float> oTensor = TensorExport.FromJson<float>(output);
                oShape = oTensor.Shape;
            }
        }

        var src = new ListTrainingDataSource<float>(iShape, oShape);
        src.AddRange(ioPairs);
        return src;
    }
}