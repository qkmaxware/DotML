using System.Security.Principal;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotML.Network;
using DotML.Network.Embedding.Text;
using DotML.Network.Initialization;
using DotML.Network.Training;

namespace DotML.Examples.InfiniteShakespeare;

public class InfiniteShakespeare : BackpropExample
{

    // For small vocab (<1k)
    /*
    FeatureChannels    = 32
    HeadHiddenSize     = 64–128
    TemporalKernelSize = 3
    WindowSize         = 5–8
    EmbeddingSize      = 16–64
    DropoutRate        = 0.05–0.1
    */

    // For medium vocabs (<10k)
    /*
    FeatureChannels    = 64
    HeadHiddenSize     = 256
    TemporalKernelSize = 3 or 5
    WindowSize         = 8–16
    EmbeddingSize      = 64–128
    DropoutRate        = 0.1–0.2
    */

    // For large vocabs (<50k)
    /*
    FeatureChannels    = 128
    HeadHiddenSize     = 512
    TemporalKernelSize = 5
    WindowSize         = 16–32
    EmbeddingSize      = 128–256
    DropoutRate        = 0.2–0.3
    */

    const int tokenWindowSize = 16;     // size of window to determine next char
    const int embeddingSize = 64;        // ideally 64-128

    public override ExampleKind Kind => ExampleKind.Generative;

    public override INetworkModule GetArchitecture()
    {
        ArchitectureFactory factory = new();
        factory.WindowSize = tokenWindowSize;
        factory.EmbeddingSize = embeddingSize;
        factory.VocabSize = LoadVocabTokens()?.Count ?? 128; // In reality, should never be null here as vocab should be created first via preprocessing

        factory.FeatureChannels = 64;        // Representation capacity
        factory.HeadHiddenSize = 256;        // control classification flexibility
        factory.TemporalKernel = 5;          // Control context modelling: try 3,5,7 etc
        factory.DropoutRate = 0.1f;
        factory.Function = ActivationFunctions.LeakyReLU;

        return factory.Make();
    }

    private LearnedDenseVectors<ShakespeareToken> GetEmbedder(List<ShakespeareToken>? tokens)
    {
        // TODO, load the vocab as ShakespeareToken objects (sorta-parsing)
        IEnumerable<ShakespeareToken> token_enum = tokens ?? Enumerable.Empty<ShakespeareToken>();

        var embedder = new LearnedDenseVectors<ShakespeareToken>(token_enum);

        if (tokens is not null && tokens.Count > 0 && File.Exists(Path.Combine(ProcessedDataPath, "embeddings.safetensors")))
        {
            Safetensors saved = Safetensors.ReadFromFile(Path.Combine(ProcessedDataPath, "embeddings.safetensors"));
            embedder.ImportEmbeddings(token_enum.Select((x, index) => (index, x)).ToDictionary((kv) => kv.x, kv => saved.GetVector<float>(kv.index.ToString())));
        }

        return embedder;
    }

    public override void ProcessRawData()
    {
        // Find all files
        var files = new DirectoryInfo(RawDataPath).EnumerateFiles()
            .Where(file => file.Extension switch
            {
                ".txt" => true,
                _ => false
            });

        // Delete old embeddings
        if (File.Exists(Path.Combine(ProcessedDataPath, "embeddings.safetensors")))
        {
            File.Delete(Path.Combine(ProcessedDataPath, "embeddings.safetensors"));
        }

        // For each file
        var parser = new Parser();
        var tokenizer = new Tokenizer();
        
        var token_ids = new Dictionary<ShakespeareToken, int>();
        var token_counts = new Dictionary<ShakespeareToken, int>();
        int total_tokens = 0;
        var ordered_tokens = new List<ShakespeareToken>();
        var all_tokens = new List<List<ShakespeareToken>>();
        Span<int> window = stackalloc int[tokenWindowSize];

        using var vocab_writer = new StreamWriter(Path.Combine(ProcessedDataPath, "vocab.csv"));
        HashSet<string> actors = new HashSet<string>();
        using var actors_writer = new StreamWriter(Path.Combine(ProcessedDataPath, "actors.txt"));
        using var training_writer = new BinaryWriter(File.Open(Path.Combine(ProcessedDataPath, "input-output.bin"), FileMode.Create));
        vocab_writer.WriteLine("id\ttoken");

        int add_token(ShakespeareToken token)
        {
            // Token already exists (add count only)
            if (token_ids.TryGetValue(token, out var id))
            {
                total_tokens++;
                if (!token_counts.TryGetValue(token, out var countd))
                {
                    countd = 0;
                }
                token_counts[token] = countd + 1;
                return id;    
            }

            // Token doesn't exist, create an id for it and add count
            id = ordered_tokens.Count;
            ordered_tokens.Add(token);
            token_ids[token] = id;
            if (!token_counts.TryGetValue(token, out var count))
            {
                count = 0;
            }
            token_counts[token] = count + 1;
            total_tokens++;
            vocab_writer.WriteLine($"{id}\t{token.ToParsableString()}");
            return id;
        }
        add_token(MissingWordToken.Instance); // Add the first, default token

        foreach (var file in files)
        {
            window.Fill(token_ids[MissingWordToken.Instance]);

            // Read all text
            using var reader = new StreamReader(file.OpenRead());

            // Parse and extract tokens
            try
            {
                var parsed = parser.Parse(reader);
                foreach (var actor in parsed.Characters)
                {
                    if (actors.Add(actor))
                    {
                        actors_writer.WriteLine(actor);
                    }
                }
                var tokens = tokenizer.Tokenize(parsed).ToList();

                // Save tokens to processed data 
                all_tokens.Add(tokens);
                foreach (var token in tokens)
                {
                    var id = add_token(token);


                    // Create record (t_1, t_2, t_3, t_4, ...) => t_next
                    // Binary format will be windowSize + 1 for the window + the predicted next token
                    // For the first token in each play this will just be (missing, ..., missing) -> first_token
                    foreach (var tokId in window) {
                        training_writer.Write(tokId);   // t_n
                    };
                    training_writer.Write(id);          // t_next

                    // Add to the window for the next token to use as prior
                    for (var i = 1; i < window.Length; i++) 
                        window[i - 1] = window[i];
                    window[window.Length - 1] = id;
                }
            } catch { continue; }
        }

        // TODO, create embedding vectors for each token type
        var embedder = this.GetEmbedder(ordered_tokens);
        var freqs = token_counts.ToDictionary(kv => kv.Key, kv => (float)kv.Value / total_tokens);
        using (var freq_writer = new StreamWriter(Path.Combine(ProcessedDataPath, "frequencies.csv")))
        {
            freq_writer.WriteLine("id\ttoken\tcount\tfrequency");
            foreach (var pair in freqs)
            {
                freq_writer.WriteLine($"{token_ids[pair.Key]}\t{pair.Key.ToParsableString()}\t{token_counts[pair.Key]}\t{pair.Value}");
            }
        }
        embedder.ImportTokenFrequencies(freqs);
        embedder.LearnEmbeddings(
            all_tokens, 
            tokenWindowSize, embeddingSize, 
            negativeSamples: 5,             // Number of negative samples per positive sample
            negativeSmoothing: 0.75f,       // Negative sampling smoothing. 0 is uniform and 1.0 is no smoothing
            range: 0.05f,                   // Vector initialization range, should be a small positive value
            learningRate: 0.02f,            // Rate of change for learned embeddings, should be a small positive value
            samplingThreshold: 1e-4f        // Token sampling threshold. 0 would indicate all tokens be dropped and 1 indicates none being dropped.
        );
        var embeddings = embedder.ExportEmbeddings();
        Safetensors sts = new Safetensors();
        foreach (var (tok, vec) in embeddings)
        {
            // Token ID used for the embedding name
            sts.Add(embedder.GetTokenId(tok).ToString(), vec);
        }
        sts.WriteToFile(Path.Combine(ProcessedDataPath, "embeddings.safetensors"));
        #pragma warning disable CS0162 // The below code is for visualization when I change the embedding size, normally this will throw a warning since embedding size > 3 for practical uses
        if (embeddingSize <= 3)
        {
            using (var pc_writer = new StreamWriter(Path.Combine(ProcessedDataPath, "blender-pointcloud.py")))
            {
                pc_writer.WriteLine(
@"import bpy

def add_point(x, y, z, 
              color=(1.0, 0.0, 0.0, 1.0), 
              label=""Point"",
              radius=0.05,
              label_offset=(0.05, 0.05, 0.05)):

    # ---- Create the point (UV Sphere) ----
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(x, y, z)
    )
    point_obj = bpy.context.active_object
    point_obj.name = f""Point_{label}""

    # ---- Create material ----
    mat = bpy.data.materials.new(name=f""Mat_{label}"")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes[""Principled BSDF""]
    bsdf.inputs[""Base Color""].default_value = color
    bsdf.inputs[""Emission Color""].default_value = color
    bsdf.inputs[""Emission Strength""].default_value = 1.0

    point_obj.data.materials.append(mat)

    return point_obj

# Point-cloud data below
"
                );
                pc_writer.WriteLine();

                foreach (var pair in embeddings)
                {
                    float x = pair.Value.ElementAtOrDefault(0), y = pair.Value.ElementAtOrDefault(1), z = pair.Value.ElementAtOrDefault(2);
                    var escaped_name = JsonSerializer.Serialize(pair.Key.ToWrittenString());
                    pc_writer.WriteLine($@"add_point({x}, {y}, {z}, label={escaped_name})");
                }
            }
        }
        #pragma warning restore CS0162
    }

    public override void LoadTrainingData(out ITrainingDataSource<float> training, out ITrainingDataSource<float> validation)
    {
        Safetensors embeddings = Safetensors.ReadFromFile(Path.Combine(ProcessedDataPath, "embeddings.safetensors"));

        var ishape = new Shape(1, tokenWindowSize, embeddingSize);
        var oshape = new Shape(embeddings.Count); // One-hot output size

        var trainingData = new ListTrainingDataSource<float>(ishape, oshape);
        var validationData = new ListTrainingDataSource<float>(ishape, oshape);
        var rng = Random.Shared;

        var path = Path.Combine(ProcessedDataPath, "input-output.bin");
        using var reader = new BinaryReader(File.OpenRead(path));
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            Tensor<float> input = Tensor<float>.Defaults(ishape);
            for (var i = 0; i < tokenWindowSize; i++)
            {
                var name = reader.ReadInt32().ToString();
                if (!embeddings.ContainsKey(name))
                    name = "0"; // missing token
                
                var embedding = embeddings.GetVector<float>(name).AsSpan();
                var row = input.SubtensorSpan(0, i);
                embedding.CopyTo(row);
            } 

            Tensor<float> output = Tensor<float>.Defaults(oshape);
            var out_ind = reader.ReadInt32();
            output[out_ind] = 1.0f; // One-hot output

            trainingData.Add((input, output));
            if (rng.NextDouble() <= 0.25)
            {
                validationData.Add((input, output));
            }
        }   
    
        training = trainingData;
        validation = validationData;
    }

    public override void ConfigureTrainer(ModuleTrainer trainer)
    {
        trainer.MaxEpochs = 500;
        trainer.LearningRateScheduler = trainer.LearningRateScheduler = new RampUpWarmup(
            maxWarmupRate: 1e-3f,
            warmupEpochs: 5,
            scheduler: new CosineAnnealing(1e-3f, trainer.MaxEpochs - 5)
        );
        trainer.BatchSize = 16;
        trainer.Initializer = new HeInitialization();
        trainer.Loss = LossFunctions.CategoricalCrossEntropy;
        trainer.Optimizer = new AdamW(weightDecay: 0.00025f);
        trainer.GlobalClipping = null;
        trainer.LocalClipping = null;
        trainer.Regularization = new NoRegularization();
        trainer.Patience = 3;
        trainer.StopCondition = static (report) => report.Metrics<AccuracyMetricsProvider>().Accuracy > 0.8f;
        trainer.Metrics.Add(new AccuracyMetricsProvider());
    }

    private static Regex tokenParser = new Regex(@"^(?<type>\w+)\((?<arg>.*)\)$", RegexOptions.Compiled);
    private List<ShakespeareToken>? LoadVocabTokens()
    {
        // Todo, load tokens from trained data
        var tokens = new List<ShakespeareToken>();
        using (var reader = new StreamReader(Path.Combine(ProcessedDataPath, "vocab.csv")))
        {
            // Skip header
            reader.ReadLine();
            var lineIndex = 1;
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();  
                if (!string.IsNullOrWhiteSpace(line))
                {
                    var parts = line.Split('\t', 2);
                    if (parts.Length != 2)
                        throw new FormatException("Missing token information on line " + lineIndex);

                    
                    var parsed = tokenParser.Match(parts[1]);
                    if (!parsed.Success)
                        throw new FormatException("Invalid vocab format on line " + lineIndex);
                    
                    switch (parsed.Groups["type"].Value)
                    {
                        case nameof(MissingWordToken):
                            tokens.Add(MissingWordToken.Instance);
                            break;
                        case nameof(ActChangeToken):
                            tokens.Add(ActChangeToken.Instance);
                            break;
                        case nameof(SceneChangeToken):
                            tokens.Add(SceneChangeToken.Instance);
                            break;
                        case nameof(StageDirectionStart):
                            tokens.Add(StageDirectionStart.Instance);
                            break;
                        case nameof(StageDirectionEnd):
                            tokens.Add(StageDirectionEnd.Instance);
                            break;

                        case nameof(SpeakerChangeToken):
                            tokens.Add(new SpeakerChangeToken(parsed.Groups["arg"].Value));
                            break;
                        case nameof(SpokenWord):
                            tokens.Add(new SpokenWord(parsed.Groups["arg"].Value));
                            break;
                        
                        case nameof(PausePunctuation):
                            tokens.Add(new PausePunctuation(parsed.Groups["arg"].Value[0]));
                            break;
                        case nameof(TerminalPunctuation):
                            tokens.Add(new TerminalPunctuation(parsed.Groups["arg"].Value[0]));
                            break;

                        default:
                            // Unknown token type
                            throw new FormatException("Unknown token type " + parsed.Groups["type"].Value + " on line " + lineIndex);
                    }
                }
                lineIndex++;
            }
        }
        return tokens;
    }

    private HashSet<string>? LoadActorsList()
    {
        // TODO, load actors from trained data
        HashSet<string> actors = new(StringComparer.InvariantCultureIgnoreCase);
        using (var reader = new StreamReader(Path.Combine(ProcessedDataPath, "actors.txt")))
        {
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (!string.IsNullOrWhiteSpace(line))
                {
                    actors.Add(line.Trim());
                }
            }
        }
        return actors;
    }

    public override void Run(IEnumerable<string> InputStrings, string? OutputPath)
    {
        // Load network
        var network = this.GetArchitecture();
        var embedder = this.GetEmbedder(LoadVocabTokens()); 
        var actors = LoadActorsList();

        // Load weights (required)
        RestoreWeights(network, throws: true);

        Console.WriteLine("Press ESC to stop generating text...");

        using TextWriter pipe = !string.IsNullOrEmpty(OutputPath) ? CreateLogger("output.txt") : System.Console.Out;

        // Create initial token span
        var tokenizer = new Tokenizer();
        var window = new ShakespeareToken[tokenWindowSize];
        Array.Fill(window, MissingWordToken.Instance);
        void push_back(ShakespeareToken token)
        {
            // Shift all tokens left
            for (var i = 1; i < window.Length; i++)
            {
                window[i-1] = window[i];
            }
            // Append to end
            window[window.Length - 1] = token;
        }
        if (InputStrings is not null && InputStrings.Any()) {
            foreach (var input in InputStrings) {
                foreach (var token in tokenizer.Tokenize(input, actors))
                {
                    push_back(token);
                }
            }
        } else
        {
            // Always start with an act change
            push_back(ActChangeToken.Instance);
            pipe.Write(ActChangeToken.Instance.ToWrittenString());
        }
    
        SpeakerChangeToken? speaker = null; bool justChangedScene = false;
        while (true)
        {
            // Stop generating
            if (Console.KeyAvailable)
            {
                if (Console.ReadKey().Key == ConsoleKey.Escape)
                    break;
            }

            // Continue generating each new token gets added to the window
            var embedding = embedder.ToTensor(window);
            embedding = embedding.ReshapeShared(embedding.Shape.NormalizeRank(3)); // Prepend a '1' bonus channel onto the tensor 1, window_length, embedding_length
            var output = network.Forward(embedding);
            var probability = new ProbabilityDistribution(new Vec<float>(output.AsArray())); // Flatten output to probability
            var selected = probability.SelectRandomly(1.3f);    // This is a number-line selection (not most probable for some randomness)
            var token = embedder.GetVocabToken(selected);   // Convert selected index to actual token
            
            // Just some formatting stuff
            if (token is SceneChangeToken changeScene)
            {
                justChangedScene = true;
            }
            else
            {
                if (token is SpeakerChangeToken changeSpeaker) {
                    speaker = changeSpeaker;
                }
                if (justChangedScene && token is not SpeakerChangeToken && token is not ActChangeToken && token is not SceneChangeToken && speaker is not null)
                {
                    pipe.Write(speaker.ToWrittenString()); // Reprint speaker if we started a new scene but continue to print text without changing speaker
                }
                justChangedScene = false;
            }   

            pipe.Write(token.ToWrittenString());
            push_back(token);
        }
    }

    public override Tensor<float> ParseUserInput(string input)
    {
        // Never gets called because Run is overwritten
        throw new NotImplementedException("Infinite Shakespeare does not support user input.");
    }

    public override string FormatOutput(string inputStr, Tensor<float> input, Tensor<float> output)
    {
        // Never gets called because Run is overwritten
        throw new NotImplementedException("Infinite Shakespeare does not support special output formatting.");
    }
}