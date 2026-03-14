using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Decodings;

/// <summary>
/// Treat the output as a json tensor
/// </summary>
public class Json : IDecoder {
    public class Result : IDecodedResult {
        private Tensor<float> tensor;

        public Result(Tensor<float> ten) {
            this.tensor = ten;
        }

        public IElement ConsoleOutput() {
            var box = new VBox();
            using var writer = new StringWriter();
            tensor.SaveJson(writer);

            box.Add(new Paragraph(writer.ToString()));
            
            return box;
        }

        public IEnumerable<FileInfo> FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                tensor.SaveJson(writer);
            }
            yield return file;
        }

        public void Dispose() { }
    }

    public IDecodedResult Decode(Tensor<float> output) {
        return new Result(
            output
        );
    }
}