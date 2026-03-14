using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Decodings;

/// <summary>
/// Treat the output as a probability distribution
/// </summary>
public class Probability : IDecoder, IWithLabels {
    public class Result : IDecodedResult {
        private ProbabilityDistribution[] dists;

        public Result(string[]? labels, Vec<float>[] vectors) {
            this.dists = vectors.Select(x => new ProbabilityDistribution(x, labels)).ToArray();
        }

        public IElement ConsoleOutput() {
            var box = new VBox();

            foreach (var dist in this.dists)
            {
                box.Add(new FixedLikelihood(dist.GetProbabilities().ToArray(), dist.GetLabels()?.ToArray()));
            }

            return box;
        }

        public IEnumerable<FileInfo> FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                foreach (var dist in this.dists) {
                    writer.Write(dist.ToString().ReplaceLineEndings());
                }
            }
            yield return file;
        }

        public void Dispose() { }
    }
    
    public string[]? Labels {get; set;}

    public IDecodedResult Decode(Tensor<float> output) {
        var batches = 1;
        for (var i = 0; i < output.Shape.Rank - 1; i++)
            batches *= output.Shape.Length(i);

        var size = output.Shape.Length(^1);

        List<Vec<float>> floats = new List<Vec<float>>();
        for (var i = 0; i < batches; i++)
            floats.Add(new Vec<float>(output.AsSpan(i * size, size).ToArray()));

        return new Result(
            this.Labels,
            floats.ToArray()
        );
    }
}