using System.Text.Json;
using DotML.Examples.MiniGpt;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Examples.Binop;

public class MiniGpt : BackpropExample
{
    public override void ConfigureTrainer(ModuleTrainer trainer)
    {
        trainer.MaxEpochs = 250;
        const int warmup = 5;
        trainer.LearningRateScheduler = new RampUpWarmup(
            maxWarmupRate: 1e-3f,
            warmupEpochs: warmup,
            scheduler: new CosineAnnealing(1e-3f, trainer.MaxEpochs - warmup)
        );
        trainer.BatchSize = 16;
        trainer.Initializer = new UniformXavierInitialization();
        trainer.Loss = LossFunctions.MeanSquaredError;
        trainer.Optimizer = new AdamW(weightDecay: 1e-3f);
        trainer.GlobalClipping = null;
        trainer.LocalClipping = null;
        trainer.Regularization = new NoRegularization(); //new L2Regularization(1e-4f);
        trainer.Patience = 3; // Patience here relates only to the stop condition below
        trainer.StopCondition = static (report) => report.Epoch > 20 && report.Loss.Average < 0.001f;
    }

    public const int WindowSize = 16;
    public const int EmbeddingSize = 16;

    public int GetVocabSize(out char[] tokens)
    {
        var path = Path.Combine(this.ProcessedDataPath, "tokens.json");
        tokens = (JsonSerializer.Deserialize<char[]>(File.ReadAllText(path)) ?? Array.Empty<char>());
        return tokens.Length + 1;
    }

    public override INetworkModule GetArchitecture()
    {
        var arch = MiniGPTArchitecture.CreateMiniGPT(
            vocab_size: GetVocabSize(out var tokens),
            n_embd: EmbeddingSize,
            block_size: WindowSize,
            n_head: 4 
        );

        return arch;
    }

    public override void ProcessRawData()
    {
        if (!Path.Exists(RawDataPath))
            Directory.CreateDirectory(RawDataPath);

        if (!Path.Exists(ProcessedDataPath))
            Directory.CreateDirectory(ProcessedDataPath);
        
        var path = Path.Combine(this.RawDataPath, "names.txt");
        var characters = File.ReadAllText(path).Where(c => !char.IsWhiteSpace(c)).Distinct().OrderBy(c => c);

        File.WriteAllText(Path.Combine(this.ProcessedDataPath, "names.text"), File.ReadAllText(path));
        File.WriteAllText(Path.Combine(this.ProcessedDataPath, "tokens.json"), JsonSerializer.Serialize(characters));
    }

    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        var token_path = Path.Combine(this.ProcessedDataPath, "tokens.json");
        var name_path = Path.Combine(this.ProcessedDataPath, "names.text");

        var tokens = JsonSerializer.Deserialize<char[]>(File.ReadAllText(token_path)) ?? Array.Empty<char>();
        var names = File.ReadAllLines(name_path).Select(name => name.Trim()).ToArray();

        var VocabSize = tokens.Length + 1;
        var BOS_INDEX = tokens.Length; // Special Beginning of string token

        var ishape = new Shape(WindowSize);
        var oshape = new Shape(VocabSize);
        ListTrainingDataSource<float> trn = new ListTrainingDataSource<float>(ishape, oshape);

        foreach (var name in names)
        {
            int[] window = new int[WindowSize];
            void push(int v)
            {
                for (var i = 1; i < window.Length; i++)
                {
                    window[i-1] = window[i];
                }
                window[window.Length - 1] = v;
            }

            Array.Fill(window, BOS_INDEX);
            foreach (var c in name)
            {
                var token_index = Array.IndexOf(tokens, c);
                if (token_index == -1) {
                    push(BOS_INDEX);
                    continue;
                }

                var input = Tensor<float>.Zeros(ishape);
                for (var i = 0; i < WindowSize; i++)
                {
                    input[i] = window[i]; // Token indices
                }

                var output = Tensor<float>.Zeros(oshape);
                output[token_index] = 1.0f;

                push(token_index);
            }

            var final_input = Tensor<float>.Zeros(ishape);
            for (var i = 0; i < WindowSize; i++)
                final_input[i] = window[i]; // Token indices
            
            var eos_output = Tensor<float>.Zeros(oshape);
            eos_output[BOS_INDEX] = 1.0f;  // Predict BOS as end marker
            
            trn.Add((final_input, eos_output));
        }

        training = trn;
        validation = trn;
    }

    public override void Run(IEnumerable<string> InputStrings, string? OutputPath)
    {
        // Load network
        var network = this.GetArchitecture();
        var vocabSize = GetVocabSize(out var tokens);
        var BOS_TOKEN = vocabSize - 1;

        // Load weights (required)
        RestoreWeights(network, throws: true);

        Console.WriteLine("Press ESC to stop generating text early...");

        using TextWriter pipe = !string.IsNullOrEmpty(OutputPath) ? CreateLogger("output.txt") : System.Console.Out;

        // Create initial token span
        var window = new int[WindowSize];
        Array.Fill(window, BOS_TOKEN);

        // Repeat until done
        var ishape = new Shape(1, WindowSize, vocabSize);
        while (true)
        {
            // Stop generating
            if (Console.KeyAvailable)
            {
                if (Console.ReadKey().Key == ConsoleKey.Escape)
                    break;
            }

            // Continue generating each new token gets added to the window
            var input = Tensor<float>.Zeros(ishape);
            for (var i = 0; i < WindowSize; i++)
            {
                input[0, i, window[i]] = 1.0f; // One hot encoding of tokens 
            }
            var output = network.Forward(input);
            var probability = new ProbabilityDistribution(new Vec<float>(output.AsArray()));
            var selected = probability.SelectRandomly(temperature: 0.5f); 
            if (selected == BOS_TOKEN || selected < 0)
                break;

            var token = tokens[selected];
            Console.Write(token);
        }
    }

    public override Tensor<float> ParseUserInput(string input)
    {
        throw new NotImplementedException("MiniGPT does not support user input.");
    }

    public override string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output)
    {
        throw new NotImplementedException("MiniGPT does not support special output formatting.");
    }
}