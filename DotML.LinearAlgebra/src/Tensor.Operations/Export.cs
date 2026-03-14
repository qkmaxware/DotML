using System.Collections;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Net;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

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

        writer.Write("\"shape\": "); writer.Write(System.Text.Json.JsonSerializer.Serialize(self.Shape.AsDimensionEnumerable())); writer.Write(',');
        writer.Write("\"dtype\": "); writer.Write(System.Text.Json.JsonSerializer.Serialize(typeof(TNum).Name)); writer.Write(',');
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
    /// Load a tensor from a JSON format such as those created via the <see cref="SaveJson"/> extension method.
    /// </summary>
    /// <param name="stream">stream containing json data</param>
    /// <returns></returns>
    /// <exception cref="NullReferenceException">thrown if mandatory fields are missing</exception>
    /// <exception cref="FormatException">thrown if the json object is incorrectly formatted</exception>
    public static Tensor<TNum> FromJson<TNum>(Stream stream)
    where TNum : INumber<TNum>
    {
        var document = JsonDocument.Parse(stream);
        var root = document.RootElement;

        return FromJson<TNum>(root);
    }
    public static Tensor<TNum> FromJson<TNum>(JsonElement root)
    where TNum : INumber<TNum>
    {
        if (!root.TryGetProperty("shape", out var shapeElement) || shapeElement.ValueKind != JsonValueKind.Array)
            throw new NullReferenceException(nameof(Shape));
        var shape = new Shape(shapeElement.EnumerateArray().Select(e => e.GetInt32()).ToArray());

        if (!root.TryGetProperty("data", out var dataElement) || dataElement.ValueKind != JsonValueKind.Array)
            throw new FormatException("Missing or invalid 'data' field");

        var tensor_elements = shape.LogicalElementCount();
        List<TNum> flat = new List<TNum>(tensor_elements); FlattenJagged(dataElement, shape, 0, flat);
        if (flat.Count != tensor_elements)
            throw new FormatException($"Jagged array contained {flat.Count} values but tensor shape expected {tensor_elements} values");

        return Tensor<TNum>.FromFlattenedArray(shape, flat.ToArray());
    }
    private static void FlattenJagged<TNum>(JsonElement? jagged, Shape shape, int dim_index, List<TNum> output)
    where TNum:INumber<TNum>
    {
        if (!jagged.HasValue)
            return;

        if (jagged.Value.ValueKind == JsonValueKind.Array)
        {
            // Is an array, recurse down
            var enumerator = jagged.Value.EnumerateArray();
            // loop over elements (if the array is smaller than the required shape, pad with 0s aka recursive null; if larger ignore extra)
            // if shape was computed properly array length should always be <= dimension length
            for (var i = 0; i < shape.Length(dim_index); i++)
            {
                if (enumerator.MoveNext())
                {
                    FlattenJagged(enumerator.Current, shape, dim_index + 1, output);
                }
                else
                {
                    FlattenJagged(null, shape, dim_index + 1, output);
                }
            }
        } else
        {
            // Is a value, try to convert to TNum
            var str = jagged.Value.GetRawText();
            try
            {
                TNum? val = (TNum?)Convert.ChangeType(str, typeof(TNum));
                output.Add(val ?? TNum.Zero);
            } catch
            {
                output.Add(TNum.Zero);
            }
        }
    }

    /*private static void ToBinary<TNum>(Tensor<TNum> self, BinaryWriter writer)
    where TNum : INumber<TNum>
    {
        // Write shape
        writer.Write(self.Rank);
        foreach (var dim in self.Shape.AsDimensionSpan())
        {
            writer.Write(dim);
        }

        // Write data-type
        if (typeof(TNum).IsAssignableTo(typeof(IConvertible)))
        {
            writer.Write((int) (((IConvertible?)default(TNum))?.GetTypeCode() ?? TypeCode.Object));
        } else
        {
            writer.Write((int)(TypeCode.Object));
        }

        // Write elements (row-major)
        writer.Write(self.ElementCount);
        foreach (var elem in self.AsSpan())
        {
            writeAction(writer, elem);
        }
    }*/

    /// <summary>
    /// Export tensor data to an Excel 2003 XML spreadsheet document. Tensor is reshaped to 3D for export.
    /// </summary>
    /// <param name="xml">xml text writer</param>
    public static void SaveSpreadsheetML<TNum>(this Tensor<TNum> self, TextWriter xml)
    where TNum : INumber<TNum>
    {
        var shape = self.Shape.EnsureRank(3);
        var tensor = self.ReshapeShared(shape);
        var spanLength = shape.Stride(^3);
        var rows = shape.Length(^2);
        var columns = shape.Length(^1);

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
        xml.WriteLine(@"    <Author>DotML</Author>");
        xml.WriteLine(@"    <Created>" + DateTime.UtcNow.ToString("s") + "Z</Created>");
        xml.WriteLine(@"  </DocumentProperties>");

        // Excel Workbook settings (optional)
        //xml.WriteLine(@"  <ExcelWorkbook xmlns=""urn:schemas-microsoft-com:office:excel"">");
        //xml.WriteLine(@"    <WindowHeight>9000</WindowHeight>");
        //xml.WriteLine(@"    <WindowWidth>13860</WindowWidth>");
        //xml.WriteLine(@"    <ProtectStructure>False</ProtectStructure>");
        //xml.WriteLine(@"    <ProtectWindows>False</ProtectWindows>");
        //xml.WriteLine(@"  </ExcelWorkbook>");

        var batchShape = shape.Slice(0..^3);

        Span<int> indices = stackalloc int[shape.Rank - 2];
        var sheets = self.ElementCount / spanLength;
        StringBuilder sb = new StringBuilder();
        for (var sheetIndex = 0; sheetIndex < sheets; sheetIndex++)
        {
            // Construct sheet name
            sb.Clear();
            for (var i = 0; i < indices.Length; i++)
            {
                if (i != 0)
                    sb.Append(',');
                sb.Append(indices[i]);
            }

            // Fetch submatrix
            var matrix = self.AsSpan2D(sheetIndex * spanLength, rows, columns);

            // Write submatrix
            xml.WriteLine(@$"  <Worksheet ss:Name=""({sb},...)"">");
            xml.WriteLine(@"    <Table>");
            for (var row = 0; row < rows; row++)
            {
                xml.WriteLine(@"      <Row>");
                for (var col = 0; col < columns; col++)
                {
                    xml.Write(@"<Cell>");
                    xml.Write(@"<Data ss:Type=""String"">");
                    xml.Write(matrix[row, col].ToString());
                    xml.Write("</Data>");
                    xml.Write(@"</Cell>");
                }
                xml.WriteLine(@"</Row>");
            }
            xml.WriteLine(@"    </Table>");
            xml.WriteLine(@"  </Worksheet>");

            // Move the next index up
            int dim = indices.Length - 1;
            while (dim >= 0)
            {
                indices[dim]++;
                if (indices[dim] < shape.Length(dim))
                    break;

                indices[dim] = 0;
                dim--;
            }
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

    /// <summary>
    /// Export tensor data to a NumPy compatible .npy file
    /// </summary>
    /// <param name="writer">npy binary writer</param>
    public static void SaveNpy(this Tensor<BFloat16> self, BinaryWriter writer)
    {
        SaveNumpyHeader(writer, self.Shape, "'float32'");
        ToJaggedArray(self, 0, 0, () => {}, () => {}, () => {}, (item) => writer.Write(item.FloatValue));
    }


    private static void SaveNumpyHeader(BinaryWriter writer, Shape Shape, string dtype)
    {
        writer.Write((byte)(93));
        writer.Write(new byte[] { (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' });
        writer.Write((byte)(1));
        writer.Write((byte)(0));

        StringBuilder sb = new StringBuilder();
        sb.Append('{');
        sb.Append("descr:");
        sb.Append(dtype);
        sb.Append(',');
        sb.Append("fortran_order: False,");
        sb.Append("shape: (");
        for (var i = 0; i < Shape.Rank; i++)
        {
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
        for (var i = 0; i < (size - bytes.Length); i++)
        {
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