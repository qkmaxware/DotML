using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Logging;


public class StatsOutputLogger : BaseOutputLogger {
    public StatsOutputLogger(DirectoryInfo logDir) : base(logDir) {
    }

    private void EmitStats(string layerName, BatchedFeatureSet<float> outputs)
    {
        for (var batch = 0; batch < outputs.Batches; batch++)
        {
            var features = outputs[batch];
            var dir_path = Path.Combine(LogDirectory.FullName, $"Batch-{batch}");
            var dir = Directory.CreateDirectory(dir_path);

            var layer_dir_path = Path.Combine(dir_path, layerName);
            var layer_dir = Directory.CreateDirectory(layer_dir_path);

            var file = Path.Combine(layer_dir_path, "statistics.csv");
            using (var writer = new StreamWriter(file))
            {
                writer.WriteLine("Metric, Value");

                var data = features.FlattenElements().ToList();
                if (!data.Any())
                    continue;

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

    public override void Visit(ConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(ConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(DepthwiseConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(DepthwiseConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(TransposeConvolutionLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(TransposeConvolutionLayer)}", args.Output);
        return;
    }

    public override void Visit(PixelShuffle layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(PixelShuffle)}", args.Output);
        return;
    }

    public override void Visit(PoolingLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(PoolingLayer)}", args.Output);
        return;
    }

    public override void Visit(FlatteningLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(FlatteningLayer)}", args.Output);
        return;
    }

    public override void Visit(DropoutLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(DropoutLayer)}", args.Output);
        return;
    }

    public override void Visit(LayerNorm layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(LayerNorm)}", args.Output);
        return;
    }

    public override void Visit(BatchNorm layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(BatchNorm)}", args.Output);
        return;
    }

    public override void Visit(DenseLinearLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(DenseLinearLayer)}", args.Output);
        return;
    }

    public override void Visit(ActivationLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(ActivationLayer)}", args.Output);
        return;
    }

    public override void Visit(SoftmaxLayer layer, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(SoftmaxLayer)}", args.Output);
        return;
    }

    public override void Visit(InputCapture capture, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        return;
    }

    public override void Visit(AdditionSkipConnection capture, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(AdditionSkipConnection)}", args.Output);
        return;
    }

    public override void Visit(ConcatenationSkipConnection capture, (int LayerIndex, BatchedFeatureSet<float> Output) args) {
        EmitStats($"Layer-{args.LayerIndex} {nameof(ConcatenationSkipConnection)}", args.Output);
        return;
    }
}