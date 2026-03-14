namespace DotML.Examples;

public interface IReport
{
    public string Name { get; }
    public string Extension { get; }
    public void Emit(TextWriter writer);
}

public class ConfusionMatrixReport: IReport
{
    private string[] labels;
    private int[,] confusion;

    public ConfusionMatrixReport(string? name, string[] labels, int[,] confusion)
    {
        this.labels = labels;
        this.Name = name ?? "confusion";
        this.confusion = confusion;
    }

    public string Name { get; private set; }
    public string Extension => ".csv";

    public void Emit(TextWriter writer)
    {
        foreach (var label in labels)
        {
            writer.Write(',');
            writer.Write(label);
        }
        writer.WriteLine();
        for (var i = 0; i < confusion.GetLength(0); i++)
        {
            writer.Write(labels[i]); 
            for (var j = 0; j < confusion.GetLength(1); j++)
            {
                writer.Write(',');
                writer.Write(confusion[i, j]);
            }
            writer.WriteLine();
        }
    }
}