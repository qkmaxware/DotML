using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;

public class Excel2003 {
    public static string Extension => ".xml";
    public static void Write(TextWriter xml, FeatureSet<double> matrices) {
        // Excel XML Header
        xml.WriteLine(@"<?xml version=""1.0""?>");
        xml.WriteLine(@"<?mso-application progid=""Excel.Sheet""?>");
        xml.WriteLine(@"<Workbook xmlns=""urn:schemas-microsoft-com:office:spreadsheet""
                            xmlns:o=""urn:schemas-microsoft-com:office:office""
                            xmlns:x=""urn:schemas-microsoft-com:office:excel""
                            xmlns:ss=""urn:schemas-microsoft-com:office:spreadsheet""
                            xmlns:html=""http://www.w3.org/TR/REC-html40"">");

        // Document Properties (optional)
        xml.WriteLine(@"  <DocumentProperties xmlns=""urn:schemas-microsoft-com:office:office"">");
        xml.WriteLine(@"    <Author>DotML Netflow</Author>");
        xml.WriteLine(@"    <Created>" + DateTime.UtcNow.ToString("s") + "Z</Created>");
        xml.WriteLine(@"  </DocumentProperties>");

        // Excel Workbook settings (optional)
        //xml.WriteLine(@"  <ExcelWorkbook xmlns=""urn:schemas-microsoft-com:office:excel"">");
        //xml.WriteLine(@"    <WindowHeight>9000</WindowHeight>");
        //xml.WriteLine(@"    <WindowWidth>13860</WindowWidth>");
        //xml.WriteLine(@"    <ProtectStructure>False</ProtectStructure>");
        //xml.WriteLine(@"    <ProtectWindows>False</ProtectWindows>");
        //xml.WriteLine(@"  </ExcelWorkbook>");

        for (var sheetIndex = 0; sheetIndex < matrices.Channels; sheetIndex++) {
            var matrix = matrices[sheetIndex];

            xml.WriteLine(@$"  <Worksheet ss:Name=""Channel-{sheetIndex}"">");
            xml.WriteLine(@"    <Table>");
            for (var row = 0; row < matrix.Rows; row++) {
                xml.WriteLine(@"      <Row>");
                for (var col = 0; col < matrix.Columns; col++) {
                    xml.Write(@"<Cell>");
                    xml.Write(@"<Data ss:Type=""Number"">");
                    xml.Write(matrix[row, col]);
                    xml.Write("</Data>");
                    xml.Write(@"</Cell>");
                }
                xml.WriteLine(@"</Row>");
            }
            xml.WriteLine(@"    </Table>");
            xml.WriteLine(@"  </Worksheet>");
        }

        // Workbook footer
        xml.WriteLine(@"</Workbook>");
    }
}

public class TensorsOutputLogger : BaseOutputLogger {

    public TensorsOutputLogger(DirectoryInfo logDir) : base(logDir) { }

    private void EmitTensors(string layerName, BatchedFeatureSet<double> args) {
        for (var batch = 0; batch < args.Batches; batch++) {
            var features = args[batch];
            var dir_path = Path.Combine(LogDirectory.FullName, $"Batch-{batch}");
            var dir = Directory.CreateDirectory(dir_path);

            var layer_dir_path = Path.Combine(dir_path, layerName);
            var layer_dir = Directory.CreateDirectory(layer_dir_path);

            var file = Path.Combine(layer_dir_path, "output.tensor" + Excel2003.Extension);
            using (var writer = new StreamWriter(file)) {
                Excel2003.Write(writer, features);
            }
        }
    }

    public override void Visit(ConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(ConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(DepthwiseConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(DepthwiseConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(TransposeConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(TransposeConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(PoolingLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(PoolingLayer)}", args.Output);
        return;
    }

    public override void Visit(FlatteningLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(FlatteningLayer)}", args.Output);
        return;
    }

    public override void Visit(DropoutLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(DropoutLayer)}", args.Output);
        return;
    }

    public override void Visit(LayerNorm layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(LayerNorm)}", args.Output);
        return;
    }

    public override void Visit(BatchNorm layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(BatchNorm)}", args.Output);
        return;
    }

    public override void Visit(DenseLinearLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(DenseLinearLayer)}", args.Output);
        return;
    }

    public override void Visit(ActivationLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(ActivationLayer)}", args.Output);
        return;
    }

    public override void Visit(SoftmaxLayer layer, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        EmitTensors($"Layer-{args.LayerIndex} {nameof(SoftmaxLayer)}", args.Output);
        return;
    }

    public override void Visit(InputCapture capture, (int LayerIndex, BatchedFeatureSet<double> Output) args) {
        return;
    }
}