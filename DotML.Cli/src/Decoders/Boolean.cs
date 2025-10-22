using System.Text;
using Qkmaxware.Terminal;
using Qkmaxware.Terminal.Elements;

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

        public IElement ConsoleOutput() {
            var builder = new StringBuilder();
            foreach (var vector in vectors)
            {
                builder.Append('[');
                var first = true;
                foreach (var value in vector)
                {
                    if (first == false)
                    {
                        builder.Append(',');
                    }
                    builder.Append(value);
                    first = false;
                }
                builder.Append(']'); builder.AppendLine();
            }
            return new Paragraph(builder.ToString());
        }

        public IEnumerable<FileInfo> FileOutput(FileInfo file) {
            using (var writer = new StreamWriter(file.OpenWrite())) {
                foreach (var vector in vectors) {
                    writer.Write('[');
                    var first = true;
                    foreach (var value in vector) {
                        if (first == false) {
                            writer.Write(',');
                        }
                        writer.Write(value);
                        first = false;
                    }
                    writer.WriteLine(']');
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