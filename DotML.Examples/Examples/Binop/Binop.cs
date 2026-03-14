using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using CommandLine;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Templates;
using DotML.Network.Training;

namespace DotML.Examples.Binop;

public class Binop : BackpropExample
{
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum OperationType
    {
        And, Or, Xor, Contradiction, Tautology, IdentityX, IdentityY, NotX, NotY, Nand, Nor, Xnor, Implication, ConverseImplication, MaterialNonImplication, ConverseNonImplication
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
        if (!TryParseConfigString<ExtraArguments>(json, out var args))
        {
            throw new FormatException("Failed to parse configuration json");
        }
        this.Operation = args!.op;
    }

    public override void TrainAllVariations(bool useExistingWeights, bool useLogging, int? saveInterval)
    {
        // Iterate over ALL operation types
        foreach (var type in Enum.GetValues<OperationType>())
        {
            this.Operation = type;
            ResetTrainingProgress();
            Train(useExistingWeights, useLogging, saveInterval);
        }
    }

    public override string? GetDescription() => "Binary operation evaluator using a neural network.";

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

    public override bool HasBeenTrained() => Enum.GetNames<OperationType>().Any((op) => File.Exists(
        Path.Combine(ExamplePath, op + "." + DefaultWeightsFilename)
    ));

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

    private bool[,] GenerateEvaluationMatrix(Func<bool, bool, bool> op)
    {
        // 0 is false, 1 is true for indices
        return new bool[,]
        {   
            // False&False, False&True
            { op(false, false), op(false, true) },
            // True&False, True&True
            { op(true, false), op(true, true) },
        };
    }
 
    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        const int TrueIndex = 1;

        // Create truth table for binary operations
        bool[,] truthTable = Operation switch
        {
            OperationType.And => GenerateEvaluationMatrix((x, y) => x && y),
            OperationType.Or => GenerateEvaluationMatrix((x, y) => x || y),
            OperationType.Xor => GenerateEvaluationMatrix((x, y) => x ^ y),
            OperationType.Contradiction => GenerateEvaluationMatrix((x, y) => false), 
            OperationType.Tautology => GenerateEvaluationMatrix((x, y) => true),
            OperationType.IdentityX => GenerateEvaluationMatrix((x, y) => x), 
            OperationType.IdentityY => GenerateEvaluationMatrix((x, y) => y), 
            OperationType.NotX => GenerateEvaluationMatrix((x, y) => !x), 
            OperationType.NotY => GenerateEvaluationMatrix((x, y) => !y), 
            OperationType.Nand => GenerateEvaluationMatrix((x, y) => !(x && y)),
            OperationType.Nor => GenerateEvaluationMatrix((x, y) => !(x || y)),
            OperationType.Xnor => GenerateEvaluationMatrix((x, y) => x == y),
            OperationType.Implication => GenerateEvaluationMatrix((x, y) => (!x)||y), 
            OperationType.ConverseImplication => GenerateEvaluationMatrix((x, y) => x||(!y)),
            OperationType.MaterialNonImplication => GenerateEvaluationMatrix((x, y) => x&&(!y)),
            OperationType.ConverseNonImplication => GenerateEvaluationMatrix((x, y) => (!x)&&y),
            _ => throw new NotSupportedException(Operation.ToString())
        };

        // Generate tensors from truth table
        const int TensorsPerOperation = 25;
        var rng = Random.Shared;
        var trueRange = Distributions.Uniform<float>(True - 0.25f, True);       // True is anything [0.25, 1]
        var falseRange = Distributions.Uniform<float>(False, False + 0.25f);    // False is anything [-1, -0.25]

        var ishape = new Shape(2);
        var oshape = new Shape(1);
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