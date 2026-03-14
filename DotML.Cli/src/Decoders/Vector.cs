using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;
using Qkmaxware.Terminal.Layout;

namespace DotML.Cli.Decodings;

/// <summary>
/// Treat the output as a raw vector
/// </summary>
public class Vector : IDecoder {
    public class Result : IDecodedResult {
        private Vec<float>[] vectors;

        public Result(Vec<float>[] vectors) {
            this.vectors = vectors;
        }

        public IElement ConsoleOutput() {
            var box = new VBox();
            foreach (var vector in vectors)
            {
                box.Add(new Paragraph(vector.ToString()));
            }
            return box;
        }

        public IEnumerable<FileInfo> FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                foreach (var vector in vectors) {
                    writer.Write(vector.ToString());
                }
            }
            yield return file;
        }

        public void Dispose() { }
    }

    public IDecodedResult Decode(Tensor<float> output) {
        var batches = 1;
        for (var i = 0; i < output.Shape.Rank - 1; i++)
            batches *= output.Shape.Length(i);

        var size = output.Shape.Length(^1);

        List<Vec<float>> floats = new List<Vec<float>>();
        for (var i = 0; i < batches; i++)
            floats.Add(new Vec<float>(output.AsSpan(i * size, size).ToArray()));

        return new Result(
            floats.ToArray()
        );
    }
}