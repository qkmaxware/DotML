using DotML.Cli.Logging;
using DotML.Network.Training;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Decodings;

/// <summary>
/// Decode an output tensor into an Excel compatible XML file
/// </summary>
public class Xml : IFileOnlyDecoder, IDecoder {

    public bool FileRequired() => false; // We have console output but it isn't preferred

    public IDecodedResult Decode(Tensor<float> output_values) {
        return new Result(output_values);
    }

    public class Result : IDecodedResult {
        private Tensor<float> values;
        public Result(Tensor<float> values) {
            this.values = values;
        }

        public IElement ConsoleOutput() {
            var box = new VBox();
            var shape = values.Shape;
            box.Add(new Label($" A {shape} tensor."));
        
            return box;
        }

        public IEnumerable<FileInfo> FileOutput(FileInfo file) {
            if (file.Extension != ".xml") {
                file = new FileInfo(file.FullName +  ".xml");
            }
            var path = file.FullName;

            using var writer = new StreamWriter(path);
            this.values.SaveSpreadsheetML(writer);

            yield return file;
        }

        public void Dispose() { }
    }
}