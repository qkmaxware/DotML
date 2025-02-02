using System.Diagnostics;
using CommandLine;
using DotML.Network;
using DotML.Network.Training;

namespace DotML.Cli.Commands;

[Verb("test", HelpText = "Test an model against a dataset")]
public class Test : BaseCommand {

    [Value(0, MetaName = "name", HelpText = "Model name", Required = true)]
    public string? ModelName {get; set;}

    [Option("data-test", HelpText = "The data-set used for testing", Required = false)]
    public string? TestingDataPath {get; set;}

    public override void Action(AppData appData) {
        var model = appData.GetModel(ModelName);
        if (model is null) {
            Console.WriteLine($"No model exists with name '{ModelName}'.");
            return;
        }
        FileInfo testing_file = new FileInfo(TestingDataPath ?? string.Empty);
        if (!testing_file.Exists) {
            Console.WriteLine($"The testing data file '{TestingDataPath}' doesn't exist.");
            return;
        }

        Console.Write("Loading model...");
        var network = model.Load();
        Console.WriteLine("done");

        Console.Write("Loading testing data...");
        var data = Fit.ReadData(testing_file);
        Console.WriteLine("done");

        Console.Write("Testing...");
        var watch = Stopwatch.StartNew();
        var report = new DefaultValidationReport();
        LossFunction loss = network.GetOutputLayer() is SoftmaxLayer 
            ? LossFunctions.CrossEntropy 
            : LossFunctions.MeanSquaredError
        ;
        Fit.Test(network, data, report, Math.Max(1, Environment.ProcessorCount), 0.1, loss);
        watch.Stop();
        var elapsed = watch.Elapsed;
        Console.WriteLine("done");

        DrawDivider();
        string[] training_headers = ["ACCURACY", "PRECISION", "RECALL", "LOSS", "VALIDATION", "TIME-TAKEN"];
        int[] training_header_len = [10,          10,          10,       10,    15,            25         ];

        for (var col = 0; col < training_headers.Length; col++) {
            var name = training_headers[col];
            var len = training_header_len[col];
            Console.Write(ColumnValue(name, len));
            Console.Write(' ');
        }
        Console.WriteLine();

        Console.Write(ColumnValue(report.Accuracy, training_header_len[0]));
        Console.Write(' '); 

        Console.Write(ColumnValue(report.Precision, training_header_len[1]));
        Console.Write(' '); 

        Console.Write(ColumnValue(report.Recall, training_header_len[2]));
        Console.Write(' '); 

        Console.Write(ColumnValue(report.AverageLoss, training_header_len[3]));
        Console.Write(' '); 

        Console.Write(ColumnValue(report.TestsPassedCount + "/" + report.TestCount, training_header_len[4]));
        Console.Write(' '); 

        Console.Write(ColumnValue(elapsed.TotalMinutes + "m", training_header_len[5]));
        Console.WriteLine();
    }
}