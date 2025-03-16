namespace DotML.Cli.Decodings;

/// <summary>
/// Treat the output as a probability distribution
/// </summary>
public class Probability : IDecoder, IWithLabels {
    public class Result : IDecodedResult {
        private ProbabilityDistribution[] dists;

        public Result(string[]? labels, Vec<double>[] vectors) {
            this.dists = vectors.Select(x => new ProbabilityDistribution(x, labels)).ToArray();
        }

        public void ConsoleOutput() {
            foreach (var dist in this.dists) {
                Console.WriteLine(dist.ToString().ReplaceLineEndings());
            }
        }

        public void FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                foreach (var dist in this.dists) {
                    writer.Write(dist.ToString().ReplaceLineEndings());
                }
            }
        }

        public void Dispose() { }
    }
    
    public string[]? Labels {get; set;}

    public IDecodedResult Decode(BatchedFeatureSet<double> output) {
        return new Result(
            this.Labels,
            output.Select(
                b => Vec<double>.Wrap(b.SelectMany(f => f.FlattenRows()).ToArray())
            ).ToArray()
        );
    }
}