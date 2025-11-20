using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using CommandLine;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Templates;
using DotML.Network.Training;

namespace DotML.Examples.Binop;

public class Binop : Example
{
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum OperationType
    {
        And, Or, Xor
    }

    public OperationType Operation = OperationType.Xor; // default to XOR
    public const float True = 1.0f;
    public const float False = -1.0f;

    public class ExtraArguments
    {
        // User can choose the binary operation they want to do
        public OperationType op { get; set; }
    }
    public override void Configure(string json)
    {
        try
        {
            var args = JsonSerializer.Deserialize<ExtraArguments>(json);
            if (args is null)
                return;

            this.Operation = args.op;
        } catch (Exception e)
        {
            throw new FormatException("Failed to parse configuration json", e);
        }
    }

    public override INetworkModule GetArchitecture()
    {
        var factory = new MultilayerPerceptronFactory();

        // Architecture (2-3-1) dense neruon network
        /*
        * I1 - H1 \
        *    x H2 - O1
        * I2 - H3 /
        */

        var settings = new MultilayerPerceptronFactory.BuildSettings();
        settings.ActivationFunction = ActivationFunctions.Tanh;
        settings.InputSize = 2;
        settings.LayerSizes = [3, 1];

        return factory.Make(settings);
    }

    public override Safetensors LoadWeights()
    {
        // IE Xor.Network.safetensors or And.Network.safetensors
        // Allows for separate weights per operation
        return Safetensors.ReadFromFile(Path.Combine(ExamplePath, Operation.ToString() + "." + DefaultWeightsFilename));
    }

    public override void SaveWeights(Safetensors tensors)
    {
        // IE Xor.Network.safetensors or And.Network.safetensors
        // Allows for separate weights per operation
        tensors.WriteToFile(Path.Combine(ExamplePath, Operation.ToString() + "." + DefaultWeightsFilename));
    }

    public override Tensor<float> ParseUserInput(string input)
    {
        // User input is expected to be a JSON array of Floats or Bools
        // ie: [-1, 1] or [false, true]

        // Float[]
        try
        {
            float[]? floats = JsonSerializer.Deserialize<float[]>(input);
            if (floats is not null)
                return Tensor<float>.Vec(floats);
        }
        catch { }

        // Bool[]
        try
        {
            bool[]? bools = JsonSerializer.Deserialize<bool[]>(input);
            if (bools is not null)
                return Tensor<float>.Vec(bools.Select(b => b ? True : False).ToArray());
        }
        catch { }

        // File path to a json file of Float[] or Bool[]
        if (File.Exists(input) && input.EndsWith(".json"))
        {
            var text = File.ReadAllText(input);

            // Float[]
            try
            {
                float[]? floats = JsonSerializer.Deserialize<float[]>(text);
                if (floats is not null)
                    return Tensor<float>.Vec(floats);
            }
            catch { }

            // Bool[]
            try
            {
                bool[]? bools = JsonSerializer.Deserialize<bool[]>(text);
                if (bools is not null)
                    return Tensor<float>.Vec(bools.Select(b => b ? True : False).ToArray());
            }
            catch { }
        }

        throw new FormatException("Unable to convert provided input to tensor");
    }

    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        const int TrueIndex = 1;

        // Create truth table for binary operations
        bool[,] truthTable = Operation switch
        {
            OperationType.And => new bool[,]
            {   
                // False&False, False&True
                { false, false },
                // True&False, True&True
                { false, true },
            },
            OperationType.Or => new bool[,]
            {   
                // False|False, False|True
                { false, true },
                // True|False, True|True
                { true, true },
            },
            OperationType.Xor => new bool[,]
            {   
                // False^False, False^True
                { false, true },
                // True^False, True^True
                { true, false },
            },
            _ => throw new NotSupportedException(Operation.ToString())
        };

        // Generate tensors from truth table
        const int TensorsPerOperation = 25;
        var rng = Random.Shared;
        var trueRange = Distributions.Uniform<float>(True - 0.25f, True);       // True is anything [0.25, 1]
        var falseRange = Distributions.Uniform<float>(False, False + 0.25f);    // False is anything [-1, -0.25]

        var ishape = new TensorShape(2);
        var oshape = new TensorShape(1);
        ListTrainingDataSource<float> trn = new ListTrainingDataSource<float>(ishape, oshape);
        ListTrainingDataSource<float> val = new ListTrainingDataSource<float>(ishape, oshape);

        float bool2float(bool b)
        {
            return b ? trueRange.Sample() : falseRange.Sample();
        }

        for (var i = 0; i < truthTable.GetLength(0); i++)
        {
            var a = i == TrueIndex ? true : false;
            for (var j = 0; j < truthTable.GetLength(1); j++)
            {
                var b = j == TrueIndex ? true : false;
                var o = truthTable[i, j];

                for (var t = 0; t < TensorsPerOperation; t++)
                {

                    Tensor<float> input = Tensor<float>.Vec([bool2float(a), bool2float(b)]);
                    Tensor<float> output = Tensor<float>.Vec([bool2float(o)]);

                    if (rng.NextDouble() > 0.25)
                    {
                        trn.Add((input, output));
                    }
                    else
                    {
                        val.Add((input, output));
                    }
                }
            }
        }

        training = trn;
        validation = val;
    }

    public override void ConfigureTrainer(ModuleTrainer trainer)
    {
        trainer.MaxEpochs = 500;
        trainer.LearningRateScheduler = new ReduceLROnPlateau(
            new RampUpWarmup(
                maxWarmupRate: 0.1f,
                warmupEpochs: 10,
                scheduler: new ConstantRate(0.1f)
            ),
            patience: 3
        );
        trainer.BatchSize = 16;
        trainer.Initializer = new UniformXavierInitialization();
        trainer.Loss = LossFunctions.MeanSquaredError;
        trainer.Optimizer = new Adam();
        trainer.Patience = 3;
        trainer.GlobalClipping = new GlobalMagnitudeClipping<float>(10);
        trainer.LocalClipping = null;
        trainer.Regularization = new NoRegularization();
        trainer.StopCondition = static (report) => report.Loss.Max < 0.2f;
    }

    public override string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output)
    {
        // [true] or [false]
        return JsonSerializer.Serialize(output.EnumerateElements().Select(x => x > 0 ? true : false));
    }
}