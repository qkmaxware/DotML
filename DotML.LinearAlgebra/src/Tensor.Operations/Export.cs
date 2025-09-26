using System.Collections;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Net;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace DotML;

/// <summary>
/// Extension methods adding export operations to supported tensor types
/// </summary>
public static class TensorExport
{
    /// <summary>
    /// Export tensor data to a JSON parseable format including shape, data-type, and array data.
    /// <example>
    /// {
    ///     "shape": [1,2,2],
    ///     "dtype": "System.Single",
    ///     "data": [[[1,2],[3,4]]]
    /// }
    /// </example
    /// </summary>
    /// <param name="writer">json text writer</param>
    public static void SaveJson<TNum>(this Tensor<TNum> self, TextWriter writer)
    where TNum : INumber<TNum>
    {
        writer.Write('{');

        writer.Write("\"shape\": "); writer.Write(System.Text.Json.JsonSerializer.Serialize(self.Shape.AsDimensionEnumerable()));
        writer.Write("\"dtype\": "); writer.Write(System.Text.Json.JsonSerializer.Serialize(typeof(TNum).Name));
        writer.Write("\"data\": "); 
        ToJaggedArray(
            self, 
            0, 0, 
            () => writer.Write('['),
            () => writer.Write(']'),
            () => writer.Write(','),
            (item) => writer.Write(System.Text.Json.JsonSerializer.Serialize(item))
        );

        writer.Write('}');
    }

    /// <summary>
    /// Export tensor data to an Excel 2003 XML spreadsheet document. Tensor is reshaped to 3D for export.
    /// </summary>
    /// <param name="xml">xml text writer</param>
    public static void SaveSpreadsheetML<TNum>(this Tensor<TNum> self, TextWriter xml)
    where TNum : INumber<TNum>
    {
        var shape = self.Shape.NormalizeRank(3);
        var tensor = self.ReshapeShared(shape);
        var channels = shape.Length(0);
        var rows = shape.Length(1);
        var columns = shape.Length(2);

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

        for (var sheetIndex = 0; sheetIndex < channels; sheetIndex++) {
            xml.WriteLine(@$"  <Worksheet ss:Name=""Channel-{sheetIndex}"">");
            xml.WriteLine(@"    <Table>");
            for (var row = 0; row < rows; row++) {
                xml.WriteLine(@"      <Row>");
                for (var col = 0; col < columns; col++) {
                    xml.Write(@"<Cell>");
                    xml.Write(@"<Data ss:Type=""String"">");
                    xml.Write(tensor[sheetIndex, row, col].ToString());
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

    /// <summary>
    /// Export tensor data to a NumPy compatible .npy file
    /// </summary>
    /// <param name="writer">npy binary writer</param>
    public static void SaveNpy(this Tensor<short> self, BinaryWriter writer)
    {
        SaveNumpyHeader(writer, self.Shape, "'int16'");
        ToJaggedArray(self, 0, 0, () => {}, () => {}, () => {}, (item) => writer.Write(item));
    }

    /// <summary>
    /// Export tensor data to a NumPy compatible .npy file
    /// </summary>
    /// <param name="writer">npy binary writer</param>
    public static void SaveNpy(this Tensor<int> self, BinaryWriter writer)
    {
        SaveNumpyHeader(writer, self.Shape, "'int32'");
        ToJaggedArray(self, 0, 0, () => {}, () => {}, () => {}, (item) => writer.Write(item));
    }

    /// <summary>
    /// Export tensor data to a NumPy compatible .npy file
    /// </summary>
    /// <param name="writer">npy binary writer</param>
    public static void SaveNpy(this Tensor<long> self, BinaryWriter writer)
    {
        SaveNumpyHeader(writer, self.Shape, "'int64'");
        ToJaggedArray(self, 0, 0, () => {}, () => {}, () => {}, (item) => writer.Write(item));
    }

    /// <summary>
    /// Export tensor data to a NumPy compatible .npy file
    /// </summary>
    /// <param name="writer">npy binary writer</param>
    public static void SaveNpy(this Tensor<Half> self, BinaryWriter writer)
    {
        SaveNumpyHeader(writer, self.Shape, "'float16'");
        ToJaggedArray(self, 0, 0, () => {}, () => {}, () => {}, (item) => writer.Write(item));
    }

    /// <summary>
    /// Export tensor data to a NumPy compatible .npy file
    /// </summary>
    /// <param name="writer">npy binary writer</param>
    public static void SaveNpy(this Tensor<float> self, BinaryWriter writer)
    {
        SaveNumpyHeader(writer, self.Shape, "'float32'");
        ToJaggedArray(self, 0, 0, () => {}, () => {}, () => {}, (item) => writer.Write(item));
    }

    /// <summary>
    /// Export tensor data to a NumPy compatible .npy file
    /// </summary>
    /// <param name="writer">npy binary writer</param>
    public static void SaveNpy(this Tensor<double> self, BinaryWriter writer)
    {
        SaveNumpyHeader(writer, self.Shape, "'float64'");
        ToJaggedArray(self, 0, 0, () => {}, () => {}, () => {}, (item) => writer.Write(item));
    }

    private static void SaveNumpyHeader(BinaryWriter writer, TensorShape Shape, string dtype) {
        writer.Write((byte)(93));
        writer.Write(new byte[]{ (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' });
        writer.Write((byte)(1));
        writer.Write((byte)(0));

        StringBuilder sb = new StringBuilder();
        sb.Append('{');
        sb.Append("descr:");
        sb.Append(dtype); 
        sb.Append(',');
        sb.Append("fortran_order: False,");
        sb.Append("shape: ("); 
        for (var i = 0; i < Shape.Rank; i++) {
            if (i != 0)
                sb.Append(',');
            sb.Append(Shape.Length(i));
        }
        sb.Append(')'); 
        sb.Append('}');
        sb.Append('\n');
        var str = sb.ToString();
        var bytes = System.Text.Encoding.ASCII.GetBytes(str);
        var size = (ushort)Math.Ceiling(bytes.Length / 64.0);
        writer.Write(size);

        writer.Write(bytes);
        for (var i = 0; i < (size - bytes.Length); i++) {
            writer.Write((byte)(20));
        }
    }


    private static void ToJaggedArray<TNum>(Tensor<TNum> self, int dim, int offset, Action openRank, Action closeRank, Action separator, Action<TNum> writeElement)
    where TNum : INumber<TNum>
    {
        var Shape = self.Shape;
        var length = Shape.Length(dim);
        openRank();
        if (dim == Shape.Rank - 1)
        {
            for (var i = 0; i < length; i++)
            {
                // Copy items
                if (i != 0)
                    separator();
                writeElement(self.AsSpan()[offset + i]);
            }
        }
        else
        {
            int stride = Shape.Stride(dim);

            for (var i = 0; i < length; i++)
            {
                // Recurse down 
                if (i != 0)
                    separator();
                var subOffset = offset + i * stride;
                ToJaggedArray(self, dim + 1, subOffset, openRank, closeRank, separator, writeElement);
            }
        }

        closeRank();
    }
}