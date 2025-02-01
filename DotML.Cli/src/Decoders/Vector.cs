namespace DotML.Cli.Decodings;

/// <summary>
/// Treat the output as a raw vector
/// </summary>
public class Vector : IDecoder {
    public class Result : IDecodedResult {
        private Vec<double> vector;

        public Result(Vec<double> vector) {
            this.vector = vector;
        }

        public void ConsoleOutput() {
            Console.WriteLine(vector.ToString());
        }

        public void FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                writer.Write(vector.ToString());
            }
        }
    }

    public IDecodedResult Decode(Vec<double> output) {
        return new Result(output);
    }
}