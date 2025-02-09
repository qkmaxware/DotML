namespace DotML.Cli.Decodings;

/// <summary>
/// Treat the output as a raw vector
/// </summary>
public class Vector : IDecoder {
    public class Result : IDecodedResult {
        private Vec<double>[] vectors;

        public Result(Vec<double>[] vectors) {
            this.vectors = vectors;
        }

        public void ConsoleOutput() {
            foreach (var vector in vectors) {
                Console.WriteLine(vector.ToString());
            }
        }

        public void FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                foreach (var vector in vectors) {
                    writer.Write(vector.ToString());
                }
            }
        }

        public void Dispose() { }
    }

    public IDecodedResult Decode(BatchedFeatureSet<double> output) {
        return new Result(
            output.Select(
                b => Vec<double>.Wrap(
                    b.SelectMany(
                        f => f.FlattenRows()
                    ).ToArray()
                )
            ).ToArray()
        );
    }
}