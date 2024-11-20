namespace DotML.Test;

/// <summary>
/// Convert ASCII art to a vector where whitespace is considered as empty and any other character is considered non-empty
/// </summary>
public class AsciiImgVectorizor : IFeatureExtractor<string> {

    public int? MaxWidth {get; set;}

    public AsciiImgVectorizor() {}

    public AsciiImgVectorizor(int width) {
        this.MaxWidth = width;
    }

    public double EmptyValue = -1.0;
    public double NonEmptyValue = 1.0;

    public Vec<double> ToVector(string value) {
        var rows = value.Split("\n");
        var row_size = rows.Length;
        var col_size = MaxWidth.HasValue ? MaxWidth.Value : rows.Select(row => row.Length).Max();
        var vector_size = row_size * col_size;

        double[] vec = new double[vector_size];
        var vector_index = 0;
        foreach (var row in rows) {
            for (var col = 0; col < col_size; col++) {
                if (col >= 0 && col < row.Length) {
                    vec[vector_index++] = char.IsWhiteSpace(row[col]) ? EmptyValue : NonEmptyValue;
                } else {
                    vec[vector_index++] = EmptyValue;
                }
            }
        }

        return Vec<double>.Wrap(vec);
    }
}