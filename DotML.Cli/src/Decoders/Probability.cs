namespace DotML.Cli.Decodings;

/// <summary>
/// Treat the output as a probability distribution
/// </summary>
public class Probability : IDecoder {
    public class Result : IDecodedResult {
        private ProbabilityDistribution dist;

        public Result(Vec<double> vector) {
            this.dist = new ProbabilityDistribution(vector);
        }

        public void ConsoleOutput() {
            Console.WriteLine(dist.ToString().ReplaceLineEndings());
        }

        public void FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                writer.Write(dist.ToString().ReplaceLineEndings());
            }
        }
    }

    public IDecodedResult Decode(Vec<double> output) {
        return new Result(output);
    }
}