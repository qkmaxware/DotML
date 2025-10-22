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

    public IDecodedResult Decode(BatchedFeatureSet<float> output) {
        return new Result(
            output.Select(
                b => Vec<float>.Wrap(
                    b.SelectMany(
                        f => f.FlattenRows()
                    ).ToArray()
                )
            ).ToArray()
        );
    }
}