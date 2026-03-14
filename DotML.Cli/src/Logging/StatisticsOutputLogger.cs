using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;


public class StatsOutputLogger : BaseOutputLogger {
    public StatsOutputLogger(DirectoryInfo logDir) : base(logDir) {
    }

    public override void Log(string identifier, Tensor<float> output)
    {
        var dir_path = Path.Combine(LogDirectory.FullName, identifier);
        var dir = Directory.CreateDirectory(dir_path);

        var file = Path.Combine(dir.FullName, "statistics.csv");
        using (var writer = new StreamWriter(file))
        {
            writer.WriteLine("Metric, Value");

            var data = output.AsSpan().ToArray().ToList(); // Clone this
            if (data.Count < 1)
                return;

            data.Sort();

            double median = 0.0;
            double mean = 0.0;
            double sum = 0.0;
            double sumSq = 0.0;

            // Compute the count
            int count = data.Count;
            writer.Write("Count,"); writer.WriteLine(count);

            // Compute min and max
            double min = data.First();
            double max = data.Last();
            writer.Write("Minimum,"); writer.WriteLine(min);
            writer.Write("Maximum,"); writer.WriteLine(max);
            writer.Write("Range,"); writer.WriteLine(max - min);

            // Compute standard deviation and variance
            for (int i = 0; i < count; i++)
            {
                double value = data[i];
                sum += value;
                sumSq += value * value;
            }
            mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            double stdDev = Math.Sqrt(variance);
            writer.Write("Standard Deviation,"); writer.WriteLine(stdDev);
            writer.Write("Variance,"); writer.WriteLine(variance);

            // Median
            if (count % 2 == 1)
            {
                median = data[count / 2];
            }
            else
            {
                median = (data[(count / 2) - 1] + data[count / 2]) / 2;
            }
            writer.Write("Sum,"); writer.WriteLine(sum);
            writer.Write("Mean,"); writer.WriteLine(mean);
            writer.Write("Median,"); writer.WriteLine(median);
            // Min, Max, Median, Mode, Standard Variation, Variance, 
        }
    }

}