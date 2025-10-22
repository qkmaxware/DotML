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

    public IDecodedResult Decode(BatchedFeatureSet<float> output) {
        return new Result(
            this.Labels,
            output.Select(
                b => Vec<float>.Wrap(b.SelectMany(f => f.FlattenRows()).ToArray())
            ).ToArray()
        );
    }
}