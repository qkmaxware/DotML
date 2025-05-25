namespace DotML.Cli.Decodings;

/// <summary>
/// Treat the output as a vector of boolean values
/// </summary>
public class BooleanVector : IDecoder {
    public class Result : IDecodedResult {
        private bool[][] vectors;

        public Result(bool[][] vectors) {
            this.vectors = vectors;
        }

        public void ConsoleOutput() {
            foreach (var vector in vectors) {
                Console.Write('[');
                var first = true;
                foreach (var value in vector) {
                    if (first == false) {
                        Console.Write(',');
                    }
                    Console.Write(value);
                    first = false;
                }
                Console.WriteLine(']');
            }
        }

        public IEnumerable<FileInfo> FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                foreach (var vector in vectors) {
                    Console.Write('[');
                    var first = true;
                    foreach (var value in vector) {
                        if (first == false) {
                            Console.Write(',');
                        }
                        Console.Write(value);
                        first = false;
                    }
                    Console.WriteLine(']');
                }
            }
            yield return file;
        }

        public void Dispose() { }
    }

    public IDecodedResult Decode(BatchedFeatureSet<float> output) {
        return new Result(
            output.Select(
                b => b.SelectMany(f => f.FlattenRows()).Select(x => x >= 0.5f ? true : false).ToArray()
            ).ToArray()
        );
    }
}