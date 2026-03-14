
using System.Diagnostics;
using System.Timers;
using DotML.Network;
using DotML.Network.Initialization;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Examples.Pong;

public class Pong : Example
{
    public override ExampleKind Kind => ExampleKind.Regression;

    public override TrainingMethod TrainingMethod => TrainingMethod.Evolutionary;

    public override void Clean()
    {
        if (Directory.Exists(ProcessedDataPath))
        {
            foreach (var file in Directory.EnumerateFiles(ProcessedDataPath))
            {
                File.Delete(file);
            }
        }
    }

    public override void Configure(string json)
    {
        // Nothing to configure   
    }
    
    protected const string DefaultWeightsFilename = "Network.safetensors";
    protected void RestoreWeights(INetworkModule network, bool throws = false)
    {
        try
        {
            var tensors = this.LoadWeights();
            var applier = new SafetensorDeserializer();
            if (network is IBlockVisitable visitable)
                applier.Deserialize(visitable, tensors);
        }
        catch (Exception)
        {
            // Re-throw if configured to
            if (throws)
                throw;
        }
    }
    protected void SaveWeights(INetworkModule network, bool throws = false)
    {
        try
        {
            var applier = new SafetensorSerializer();
            if (network is IBlockVisitable visitable)
                applier.Serialize(visitable);
            this.SaveWeights(applier.ToSafetensors());
        }
        catch (Exception)
        {
            // Re-throw if configured to
            if (throws)
                throw;
        }
    }
    public virtual void SaveWeights(Safetensors tensors)
    {
        // Ensure none of the weights are NaN or invalid values
        static double value2Double(object? obj)
        {
            if (obj is null)
                return double.NaN;

            return Convert.ToDouble(obj);
        }
        static bool isInvalid(double val)
        {
            return double.IsNaN(val);
        }
        foreach (var key in tensors.Keys())
        {
            if (tensors.AnyIn(key, (v) => { double val = value2Double(v); return isInvalid(val); }))
            {
                throw new ArgumentException(key, "NaN values found in model weights");
            }
        }
        // Save the weights
        tensors.WriteToFile(Path.Combine(ExamplePath, DefaultWeightsFilename));
    }
    public virtual Safetensors LoadWeights()
    {
        return Safetensors.ReadFromFile(Path.Combine(ExamplePath, DefaultWeightsFilename));
    }

    public override bool HasBeenTrained()
    {
        return File.Exists(Path.Combine(ExamplePath, DefaultWeightsFilename));
    }

    private static readonly TimeSpan dt = TimeSpan.FromSeconds(1.0f / 30.0f);

    public override void Run(IEnumerable<string> inputs, string? outputPath)
    {
        Console.CursorVisible = false;
        Game game = new Game(width: 120, height: 40);
        var scene = Game.Mode.Menu;

        var ctrlHumanP1 = new HumanController(true);
        //var ctrlAiP1 = new SimpleAiController(game, true);
        var ctrlAiP1NN = new NNAiController(game, true);
        var ctrlHumanP2 = new HumanController(false);
        //var ctrlAiP2 = new SimpleAiController(game, false);
        var ctrlAiP2NN = new NNAiController(game, false);

        var timer = new System.Timers.Timer(dt);
        Game.FrameEvents evts;
        timer.Elapsed += (Object? source, ElapsedEventArgs e) =>
        {
            switch (scene)
            {
                case Game.Mode.Menu:
                    scene = game.LoopMenu(dt, ctrlHumanP1, ctrlHumanP2);
                    if (scene != Game.Mode.Menu || scene != Game.Mode.VsHuman)
                    {
                        // If AI play selected, reload the networks
                        RestoreWeights(ctrlAiP1NN.Network);
                        RestoreWeights(ctrlAiP2NN.Network);
                    }
                    break;
                case Game.Mode.VsHuman:
                    game.LoopGame(dt, ctrlHumanP1, ctrlHumanP2, out evts);
                    break;
                case Game.Mode.VsAI:
                    game.LoopGame(dt, ctrlHumanP1, ctrlAiP2NN, out evts);
                    break;
                case Game.Mode.AiOnly:
                    game.LoopGame(dt, ctrlAiP1NN, ctrlAiP2NN, out evts);
                    break;
            }
            game.FlushBuffer();
        };
        timer.AutoReset = true;
        timer.Start();

        // Spin wait while playing
        while (timer.Enabled)
        {
            Thread.Sleep(500);
        }
        Console.CursorVisible = true;
    }

    public override void Train(bool useExistingWeights, bool useLogging, int? saveInterval)
    {
        var initializer = new NormalXavierInitialization();
        var EMPTY = Tensor<float>.Vec([ 0 ]);

        GeneticTrainer trainer = new GeneticTrainer();
        trainer.PopulationSize = 1_000;
        trainer.BatchSize = Environment.ProcessorCount;
        var max_generations = 30;
        trainer.MaxGenerations = max_generations;
        trainer.Factory = () =>
        {
            // Make a genome for the network
            var network = (SequentialBlock)((ArchitectureBlock)NNAiController.MakeNetwork()).RootModule;
            network.Initialize(initializer);
            var wbcount = 0;
            List<Tensor<float>> weights = new(network.LayerCount);
            List<Tensor<float>> biases = new(network.LayerCount);
            foreach (var layer in network.AsEnumerable())
            {
                if (layer is not IWeightsAndBiasNetworkModule wb)
                {
                    weights.Add(EMPTY);
                    biases.Add(EMPTY);
                    continue;
                }   

                weights.Add(wb.Weights);
                biases.Add(wb.Biases);
                wbcount++;
            }
            if (wbcount == 0)
                throw new Exception("No weights encoded");
            return new WeightsAndBiasGenome(weights, biases);
        };
        trainer.FitnessTest = new PongFitnessTest() { MaxGenerations = max_generations };
        trainer.Proportions = new PopulationProportion(elite: 25, crossover: 50, eliteMutations: 20, random: 5);
        trainer.StopCondition = null; // TODO?

        Console.WriteLine("Info:");
        var templateNetwork = NNAiController.MakeNetwork();
        var networkName = templateNetwork is ArchitectureBlock nameArch ? nameArch.Name : templateNetwork.GetType().Name;
        Console.WriteLine($"  Example: {this.Name}");
        Console.WriteLine($"  Network: {networkName}");
        Console.WriteLine($"  Population Size: {trainer.PopulationSize}");
        Console.WriteLine($"  Population Spread: {(trainer.Proportions.PercentElite * 100):F1}% elite, {(trainer.Proportions.PercentCrossover * 100):F1}% crossover, {(trainer.Proportions.PercentMutation * 100):F1}% mutation, {(trainer.Proportions.PercentRandom * 100):F1}% random");
        Console.WriteLine($"  Max Generations: {trainer.MaxGenerations}");
        Console.WriteLine();

        Console.WriteLine("Training...");

        var session = trainer.EnumerateTraining();
        WriteRow("Generation", "Avg Fitness", "Max Fitness", "Min Fitness", "Time");
        Stopwatch watch = Stopwatch.StartNew();
        while (session.MoveNext())
        {
            watch.Stop();
            var report = session.Current;
            WriteRow(report.Generation, report.Fitness.Average, report.Fitness.Max, report.Fitness.Min, watch.Elapsed);

            if (saveInterval.HasValue && saveInterval.Value > 0 && report.Generation != 0 && report.Generation % saveInterval.Value == 0)
            {
                var best1 = session.Current.MostFit!;
                var network1 = ToNetwork(best1);

                SaveWeights(network1);
                Console.WriteLine("Saved weights");
            }
            ((PongFitnessTest)trainer.FitnessTest).Generation++;
            watch.Restart();
        }

        var best = session.Current.MostFit!;
        var network = ToNetwork(best);

        SaveWeights(network);
        Console.WriteLine("Saved weights");
    }

    private static INetworkModule ToNetwork(IGenome genome)
    {
        if (genome is not WeightsAndBiasGenome wb)
                throw new ArgumentException(nameof(genome), "Type not supported");

        var network = (ArchitectureBlock)NNAiController.MakeNetwork();
        var list = (SequentialBlock)network.RootModule;

        for (var i = 0; i < list.LayerCount; i++)
        {
            var layer = list.GetLayer(i) as DenseLinear;
            if (layer is null)
                continue;

            layer.Weights = wb.GetWeight(i);
            layer.Biases = wb.GetBias(i);
        }
        return network;
    }

    private class PongFitnessTest : ParallelTestScheduler
    {
        public int Generation = 0; // Incremented externally at end of generation
        public int MaxGenerations = 100;
        public override float Test(IGenome genome)
        {
            // Regenerate the network from genome
            var network = ToNetwork(genome);
            
            // Create game
            Game game = new Game(width: 120, height: 40);
            game.ResetGame();

            IController ctrl1 = new NNAiController(game, isP1: true, network: network); // NN controls actions taken
            IController ctrl2 = new SimpleAiController(game, isP1: false);              // Tracks up and down automatically
            float difficulty = (float)Generation / (float)MaxGenerations;               // Difficulty starts at 0 and goes till 1
            game.Right.Speed = (1 - difficulty) * 0 + difficulty * game.Left.Speed;     // Other paddle starts slow and increases to normal speed over time

            // Simulate game (not realtime)
            TimeSpan elapsed = TimeSpan.Zero;
            TimeSpan maxTime = TimeSpan.FromMinutes(15);

            Game.FrameEvents evts;
            int hits = 0,                       // Number of times I hit the ball
            framesAlive = 0;                    // Number of frames that the ball is alive for
            Metric<float> missDistance = new(); // Avg, Max, Min distance between paddle and ball when they score on me
            while (elapsed < maxTime)
            {
                elapsed += dt;
                game.LoopGame(dt, ctrl1, ctrl2, out evts);

                // Handle frame events
                hits += evts.LeftReturned ? 1 : 0;
                var alive = !evts.LeftScored || !evts.RightScored;
                framesAlive += alive ? 1 : 0;
                if (evts.RightScored)
                {
                    missDistance.AddSample(evts.MissedItBy);
                }
            }

            // Compute fitness of AI (more fit means more points than the opponent)
            var p1Score = game.Left.Score;
            var p2Score = game.Right.Score;

            var scoreDifference = p1Score - p2Score; // Will be large positive if dominating, large negative if being dominated
            
            float fitness = 
                  5f * hits
                + 50f * scoreDifference
                //+ 0.1f * framesAlive (not sure if this is good to add because it doesn't differentiate between idle frames because of getting a goal or getting scored on)
                - 0.5f * missDistance.Average
            ;

            return fitness;
        }
    }

    public override void Validate(bool useLogging)
    {
        Game game = new Game(width: 120, height: 40);
        game.ResetGame();

        var ctrlAiP1NN = new NNAiController(game, true);
        var ctrlAiP2NN = new SimpleAiController(game, false);

        RestoreWeights(ctrlAiP1NN.Network);

        Console.CursorVisible = false;

        const int speedFactor = 4;

        while (game.Left.Score < 5)
        {
            game.LoopGame(dt, ctrlAiP1NN, ctrlAiP2NN, out var evts);
            game.FlushBuffer();
            Thread.Sleep(dt / speedFactor);
        }

        Console.CursorVisible = true;
    }
}