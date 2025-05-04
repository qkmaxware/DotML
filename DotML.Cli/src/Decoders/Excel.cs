using DotML.Cli.Logging;
using DotML.Network.Training;

namespace DotML.Cli.Decodings;

/// <summary>
/// Decode an output tensor into an Excel compatible XML file
/// </summary>
public class Xml : IFileOnlyDecoder, IDecoder {

    public bool FileRequired() => false; // We have console output but it isn't preferred

    public IDecodedResult Decode(BatchedFeatureSet<double> output_values) {
        return new Result(output_values);
    }

    public class Result : IDecodedResult {
        private BatchedFeatureSet<double> values;
        public Result(BatchedFeatureSet<double> values) {
            this.values = values;
        }

        public void ConsoleOutput() {
            int i = 1;
            foreach (var tensor in values) {
                var shape = tensor.Shape;
                Console.WriteLine($"Batch {i++}: A {shape.Channels}x{shape.Rows}x{shape.Columns} tensor.");
            }
        }

        public void FileOutput(FileInfo file) {
            if (file.Extension != Excel2003.Extension) {
                file = new FileInfo(file.FullName + Excel2003.Extension);
            }
            var path = file.FullName;

            if (this.values.Batches == 1) {
                using var writer = new StreamWriter(path);
                Excel2003.Write(writer, this.values[0]);
            } else {
                var batch_id = 0;
                foreach (var features in this.values) {
                    var name = Path.ChangeExtension(path, $".{batch_id}{Excel2003.Extension}");
                    using var writer = new StreamWriter(path);
                    Excel2003.Write(writer, features);
                }
            }
        }

        public void Dispose() { }
    }
}