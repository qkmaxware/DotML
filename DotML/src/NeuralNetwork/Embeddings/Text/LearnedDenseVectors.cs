namespace DotML.Network.Embedding.Text;

/// <summary>
/// Dense-vector embedding where each token maps to a separate dense vector. It can additionally add positional encoding to the output vectors. Rather than <see cref="DenseVectors"/> these embeddings are not provided directly by the user, but instead are learned using skip-grams with both positive and negative sampling. Before use one should call <see cref="LearnTokenFrequencies"/> and <see cref="LearnEmbeddings"/> in that order or simply import the Embeddings using <see cref="ImportEmbeddings"/>.
/// </summary>
/// <typeparam name="TToken">token type</typeparam>
public class LearnedDenseVectors<TToken> : IEmbedding<IEnumerable<TToken>, float> where TToken:notnull
{
    private Dictionary<TToken, int> Token2Id;
    private Dictionary<int, TToken> Id2Token;
    private Dictionary<TToken, float> Token2Freq;

    private int EmbeddingVectorLength;
    private Dictionary<TToken, Vec<float>>? TokenEmbeddings = null;
    
    public TToken GetVocabToken(int id)
    {
        return Id2Token[id];
    }

    public int GetTokenId(TToken token)
    {
        return Token2Id[token];
    }

    /// <summary>
    /// Create a new embedding using the provided vocabulary of tokens
    /// </summary>
    /// <param name="vocab">All tokens in the vocabulary</param>
    public LearnedDenseVectors(params IEnumerable<TToken> vocab)
    {
        this.Token2Id = vocab.Distinct().Index().ToDictionary((x) => x.Item, (x) => x.Index); 
        this.Id2Token = this.Token2Id.ToDictionary((kv) => kv.Value, (kv) => kv.Key);
        var one_over = 1.0f/this.Token2Id.Count;  
        this.Token2Freq = this.Token2Id.ToDictionary((kv) => kv.Key, (kv) => one_over); // Default to uniform. Can add freqs later using LearnTokenFrequencies
    }

    /// <summary>
    /// Pre-compute the relative frequencies of each token type. Tokens missing from the vocab will have their frequency set to a very small positive value rather than 0.
    /// </summary>
    /// <param name="stream">Tokenized stream from corpus. Should contain ALL tokens within your vocab.</param>
    public void LearnTokenFrequencies(IEnumerable<TToken> stream)
    {
        var tokens = 0;
        Dictionary<TToken, int> counts = new Dictionary<TToken, int>();

        foreach (var tok in stream)
        {
            if (!Token2Id.ContainsKey(tok))
                continue;

            int count;
            if (!counts.TryGetValue(tok, out count))
                count = 0;

            counts[tok] = count + 1;
            tokens++;
        }
        if (tokens == 0)
            throw new ArgumentException(nameof(stream), "Token stream is empty");

        var updatedFreq = new Dictionary<TToken, float>();
        foreach (var tok in Token2Id.Keys)
        {
            if (counts.TryGetValue(tok, out var count))
            {
                updatedFreq[tok] = (float)count / tokens;
            } else
            {
                const float epsilon = 1e-5f;
                updatedFreq[tok] = epsilon; // Rather than 0, give it a very small freq
            }
        }
        this.Token2Freq = updatedFreq;
    }

    /// <summary>
    /// Import pre-computed frequencies for each token. All token types must have a provided frequency
    /// </summary>
    /// <param name="frequencies">frequencies</param>
    /// <exception cref="ArgumentException">thrown if some tokens are missing frequencies</exception>
    public void ImportTokenFrequencies(Dictionary<TToken, float> frequencies)
    {
        var nextDict = new Dictionary<TToken, float>();
        foreach (var token in this.Token2Id.Keys)
        {
            if (frequencies.TryGetValue(token, out var freq))
            {
                nextDict[token] = freq;
            }
            else
            {
                throw new ArgumentException(nameof(frequencies), "Missing frequencies for one or more tokens");
            }
        }
        this.Token2Freq = nextDict;
    }

    /// <summary>
    /// Perform iterative learning to discover the embedding vectors for each token
    /// </summary>
    /// <param name="stream">Tokenized stream from a single corpus.</param>
    /// <param name="windowSize">Sampling window size.</param>
    /// <param name="embeddingSize">Length of each embedding vector.</param>
    /// <param name="negativeSamples">Number of negative samples to take. 0 is strictly softmas training (default 0).</param>
    /// <param name="negativeSmoothing">Negative sampling smoothing. 0 is uniform and 1.0 is no smoothing (default 0.75).</param>
    /// <param name="range">Vector initialization range, should be a small positive value (default 0.01).</param>
    /// <param name="learningRate">Rate of change for learned embeddings, should be a small positive value (default 0.01).</param>
    /// <param name="samplingThreshold">Token sampling threshold. 0 would indicate all tokens be dropped and 1 indicates none being dropped. Small values of around 1e-5 are considered good for dropping frequent, but contextually irrelevant tokens (default 1.0).</param>
    public void LearnEmbeddings(IEnumerable<TToken> stream, int windowSize, int embeddingSize, int negativeSamples = 0, float negativeSmoothing = 0.75f, float range = 0.01f, float learningRate = 0.01f, float samplingThreshold = 1.0f)
    {
        LearnEmbeddings([stream], windowSize, embeddingSize, negativeSamples, negativeSmoothing, range, learningRate, samplingThreshold);
    }

    /// <summary>
    /// Perform iterative learning to discover the embedding vectors for each token
    /// </summary>
    /// <param name="stream">Tokenized stream from multiple corpus.</param>
    /// <param name="windowSize">Sampling window size.</param>
    /// <param name="embeddingSize">Length of each embedding vector.</param>
    /// <param name="negativeSamples">Number of negative samples to take. 0 is strictly softmas training (default 0).</param>
    /// <param name="negativeSmoothing">Negative sampling smoothing. 0 is uniform and 1.0 is no smoothing (default 0.75).</param>
    /// <param name="range">Vector initialization range, should be a small positive value (default 0.01).</param>
    /// <param name="learningRate">Rate of change for learned embeddings, should be a small positive value (default 0.01).</param>
    /// <param name="samplingThreshold">Token sampling threshold. 0 would indicate all tokens be dropped and 1 indicates none being dropped. Small values of around 1e-5 are considered good for dropping frequent, but contextually irrelevant tokens (default 1.0).</param>
    public void LearnEmbeddings(IEnumerable<IEnumerable<TToken>> streams, int windowSize, int embeddingSize, int negativeSamples = 0, float negativeSmoothing = 0.75f, float range = 0.01f, float learningRate = 0.01f, float samplingThreshold = 1.0f)
    {
        // Sanitize args
        windowSize = Math.Max(1, windowSize);
        embeddingSize = Math.Max(1, embeddingSize);
        range = Math.Abs(range);
        negativeSamples = Math.Max(0, negativeSamples);
        learningRate = Math.Abs(learningRate);
        negativeSmoothing = Math.Clamp(negativeSmoothing, 0.0f, 1.0f); // Default value of 0.75 is decent smoothing, use 1 for no smoothing and 0 for uniform dist
        samplingThreshold = Math.Clamp(samplingThreshold, 0.0f, 1.0f); // Default of 1 is no dropout, a good in-practice value should be 1e-5 though

        // Initialize random vectors
        var random = Distributions.Uniform<float>(-range, range);

        Dictionary<TToken, Vec<float>> E = new Dictionary<TToken, Vec<float>>();
        Dictionary<TToken, Vec<float>> O = new Dictionary<TToken, Vec<float>>();
        foreach (var token in Token2Id.Keys)
        {
            E[token] = random_vec(random, embeddingSize);
            O[token] = random_vec(random, embeddingSize);
        }

        // Create selection weighting table
        float weightSum = 0;
        Dictionary<TToken, float> selectionWeights = new Dictionary<TToken, float>();
        Dictionary<TToken, float> dropProbabilities = new Dictionary<TToken, float>();
        foreach (var (tok, freq) in this.Token2Freq)
        {
            var weight = MathF.Pow(freq, negativeSmoothing);
            selectionWeights[tok] = weight;
            weightSum += weight;

            dropProbabilities[tok] = Math.Clamp(1.0f - MathF.Sqrt(samplingThreshold / freq), 0.0f, 1.0f);
        }

        // Create streaming window
        TToken[] buffer = new TToken[2 * windowSize + 1];
        foreach (var stream in streams) {
            Array.Fill(buffer, default(TToken));
            int indexOfFirstValid = buffer.Length - 1; // set to the last spot index since this is where the first push will put a valid token

            // Begin streaming 
            foreach (var nextToken in stream)
            {   
                // Skip token if its not in the vocab
                if (!Token2Id.ContainsKey(nextToken))
                    continue;

                // Subsampling
                var dropProbability = dropProbabilities[nextToken];
                if (Random.Shared.NextSingle() < dropProbability)
                    continue;

                // Add to window (push prior tokens left)
                for (var i = 1; i < buffer.Length; i++)
                    buffer[i - 1] = buffer[i];
                buffer[buffer.Length - 1] = nextToken;
                if (indexOfFirstValid > 0)
                {
                    // Window is not filled, continue until filled
                    indexOfFirstValid--;
                    continue;
                }

                // Handle learning
                var centre = buffer[windowSize];

                for (var i = 0; i < buffer.Length; i++)
                {
                    if (i == windowSize)    
                        continue;
                    
                    var context = buffer[i];
                    train_pair(centre, context, E, O, learningRate, embeddingSize, negativeSamples, selectionWeights, weightSum);
                }
            }
        }

        this.TokenEmbeddings = E;
        EmbeddingVectorLength = embeddingSize;
    }
    
    /// <summary>
    /// Import a saved vector embedding for each token
    /// </summary>
    /// <param name="embeddings">embeddings to import. An embedding vector must be provided for each vocab token and all vectors are required to be the same length</param>
    /// <exception cref="ArgumentException">thrown if vectors are of mismatched size or a vector is missing for any tokens</exception>
    public void ImportEmbeddings(IDictionary<TToken, Vec<float>> embeddings)
    {
        var nextEmbeddings = new Dictionary<TToken, Vec<float>>();
        int embeddingLength = 0;
        foreach (var token in Token2Id.Keys)
        {
            if (embeddings.TryGetValue(token, out var vec))
            {
                if (nextEmbeddings.Count != 0 && embeddingLength != vec.Dimensionality)
                    throw new ArgumentException(nameof(embeddings), "All embedding vectors must have the same dimensionality");
                nextEmbeddings[token] = vec;
                embeddingLength = vec.Dimensionality;
            } else
            {
                throw new ArgumentException(nameof(embeddings), $"Missing embedding for token {token}. All tokens must have a provided embedding");
            }
        }

        this.EmbeddingVectorLength = embeddingLength;
        this.TokenEmbeddings = nextEmbeddings;
    }

    /// <summary>
    /// Export all learned vector embeddings for each token 
    /// </summary>
    /// <returns>dictionary of token to embedding vector</returns>
    public Dictionary<TToken, Vec<float>> ExportEmbeddings()
    {
        if (this.TokenEmbeddings is null)
            return new Dictionary<TToken, Vec<float>>();
        return this.TokenEmbeddings.ToDictionary((kv) => kv.Key, (kv) => kv.Value.Clone());
    }

    private static Vec<float> random_vec(IProbabilityDistribution<float> random, int d)
    {
        Vec<float> vec = new Vec<float>(Math.Max(0, d));
        for (var i = 0; i < vec.Dimensionality; i++)
        {
            vec[i] = random.Sample();
        }
        return vec;
    }
    private float sigmoid(float x)
    {
        return 1.0f / (1.0f + MathF.Exp(-x));
    }
    private void train_pair(TToken centreTok, TToken contextTok, Dictionary<TToken, Vec<float>> E, Dictionary<TToken, Vec<float>> O, float learningRate, int d, int numNegative, Dictionary<TToken, float> selectionWeights, float totalWeighting)
    {
        // Positive sample
        var centreVec = E[centreTok];
        var contextVec = O[contextTok];

        var score = centreVec.Dot(contextVec);
        var prob = sigmoid(score);
        var error = 1 - prob;

        for (var i = 0; i < d; i++)
        {
            var in_i = @centreVec[i];
            var out_i = @contextVec[i];

            @centreVec[i] += learningRate * error * out_i;
            @contextVec[i] +=  learningRate * error * in_i;
        }

        // Negative samples
        for (var negSample = 0; negSample < numNegative; negSample++)
        {
            // Try to get negative sample different from centre token
            int tries = 5;
            TToken? negativeTok;
            Vec<float> negativeVec;
            do {
                negativeTok = randomVocabToken(selectionWeights, totalWeighting);
                negativeVec = O[negativeTok];
                tries--;
            } while ((negativeTok.Equals(contextTok) || negativeTok.Equals(centreTok)) && tries > 0);
            if (negativeTok.Equals(contextTok) || negativeTok.Equals(centreTok))
                continue; 

            var scoreNeg = centreVec.Dot(negativeVec);
            var probNeg = sigmoid(scoreNeg);
            var errorNeg = 0 - probNeg;

            for (var i = 0; i < d; i++)
            {
                var in_i = @centreVec[i];
                var out_i = @negativeVec[i];

                @centreVec[i] += learningRate * errorNeg * out_i;
                @negativeVec[i] +=  learningRate * errorNeg * in_i;
            }
        }
    }

    private TToken randomVocabToken(Dictionary<TToken, float> selectionWeights, float totalWeighting)
    {
        // TODO sample with frequency of token (number-line sampling)
        var sample = Random.Shared.NextSingle();

        var baseProbability = 0.0f; 
        foreach (var (token, weight) in selectionWeights)
        {
            var probOfSelection = weight / totalWeighting;
            if (sample <= baseProbability + probOfSelection)
            {
                // Pick me!!!
                return token;
            }

            baseProbability += probOfSelection;
        }

        // Fallback, pick uniformly
        var negativeId = Random.Shared.Next(0, Token2Id.Count);
        return Id2Token[negativeId];
    }

    public IPositionEncoder? PositionEncoder {get; set;}

    /// <summary>
    /// Convert a value to a tensor embedding of shape [len(sequence), len(embedding_vector)]
    /// </summary>
    /// <param name="value">value to convert</param>
    /// <returns>tensor representation of the value</returns>
    public Tensor<float> ToTensor(IEnumerable<TToken> value)
    {
        if (TokenEmbeddings is null || TokenEmbeddings.Count == 0)
            throw new NullReferenceException($"{nameof(TokenEmbeddings)} is null, {nameof(LearnEmbeddings)} on a corpus of tokens or {nameof(ImportEmbeddings)} must be called before an embedding can be created.");

        var toks = value.ToList();

        var concatLength = this.PositionEncoder?.PositionEncodingLength ?? 0;
        var shape = new TensorShape(toks.Count, EmbeddingVectorLength + concatLength);
        var embedding = Tensor<float>.Zeros(shape);
        int row = 0;
        foreach (var tok in toks)
        {
            if (!TokenEmbeddings.TryGetValue(tok, out var vec))
            {
                row++;
                continue;
            }

            var embedding_row = embedding.SubtensorSpan(row);
            var vec_row = vec.AsSpan();
            vec_row.CopyTo(embedding_row);

            PositionEncoder?.EmbedPosition(row, embedding_row);

            row++;
            continue;
        }

        return embedding;
    }
}