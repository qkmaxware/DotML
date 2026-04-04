using DotML.Network;
using DotML.Network.Embedding.Text;
using DotML.Network.Initialization;
using DotML.Network.Training;

/*
### Step 1: Create Network
```csharp
INetworkModule network = MiniGPTArchitecture.CreateMiniGPT(
    vocab_size: 27,      // Your vocabulary size
    n_embd: 16,          // [Optional] embedding dimension
    block_size: 16,      // [Optional] context length
    n_head: 4            // [Optional] attention heads
);
```

### Step 2: Initialize
```csharp
network.Initialize(new XavierUniform());
```

### Step 3: Forward Pass
```csharp
var input = CreateOneHotBatch(8, 16, 27);  // [batch, seq, vocab]
var logits = network.Forward(input);        // [batch, seq, vocab]
```

### Step 4: Use Output
```csharp
// For training: compute loss and backprop
var loss = ComputeLoss(logits, targets);

// For inference: sample or greedy
var next_token = argmax(logits[0, -1, :]);
```
*/

namespace DotML.Examples.MiniGpt;

/// <summary>
/// Mini-GPT architecture converted from Python to DotML.
/// 
/// Original Python Architecture:
/// - Token embedding: vocab_size -> n_embd (16)
/// - Position embedding: block_size (16) -> n_embd (16)  
/// - 1 transformer layer with:
///   - Multi-head self-attention (4 heads, d_k=4)
///   - Feed-forward MLP (n_embd -> 4*n_embd -> n_embd) with ReLU
///   - Residual connections around both sub-blocks
/// - RMSNorm after each sub-block
/// - Output head: n_embd -> vocab_size
/// 
/// DotML Implementation Notes:
/// - Token and position embeddings are implemented as DenseLinear layers
/// - Assumes input is [batch, block_size, vocab_size] as one-hot encoded tokens
/// - RMSNorm is approximated with LayerNorm
/// - ResidualAdd combines blocks with identity skip connections
/// </summary>
public static class MiniGPTArchitecture
{
    /// <summary>
    /// Creates the Mini-GPT network architecture.
    /// 
    /// Expected input shape: [batch_size, block_size, vocab_size] as one-hot encoded tokens
    /// Output shape: [batch_size, block_size, vocab_size] logits
    /// </summary>
    /// <param name="vocab_size">Size of the vocabulary including BOS token</param>
    /// <param name="n_embd">Embedding dimension (default 16)</param>
    /// <param name="block_size">Context window size (default 16)</param>
    /// <param name="n_head">Number of attention heads (default 4)</param>
    /// <returns>A complete INetworkModule representing the Mini-GPT</returns>
    public static INetworkModule CreateMiniGPT(
        int vocab_size,
        int n_embd = 16,
        int block_size = 16,
        int n_head = 4)
    {
        if (n_embd % n_head != 0)
            throw new ArgumentException("n_embd must be divisible by n_head", nameof(n_embd));

        int d_k = n_embd / n_head;
        int mlp_hidden = 4 * n_embd;

        // ===== Embedding Stage =====
        // Token embedding: vocab_size -> [block_size, n_embd]
        // Combine token and position embeddings
        var embedding_combine = new LearnedEmbedding(vocabSize: vocab_size, maxSeqLen: block_size, embeddingDim: n_embd);

        // ===== Transformer Block =====
        // Self-attention [B, T, n_embd] -> [B,n_head,T,d_k]
        var self_attention = new SelfAttention(d_model: n_embd, d_k: d_k, heads: n_head);

        // Normalization + Attention (pre-norm style)
        var attention_normalized = new SequentialBlock(new INetworkModule[]
        {
            new LayerNorm(n_embd), // Is this the right "normalized shape" to use for LayerNorm
            self_attention
        });

        // Residual: Attention + skip
        var attention_residual = new ResidualAdd(attention_normalized, new Identity());

        // ===== MLP Block =====
        // Hidden layer: n_embd -> mlp_hidden with ReLU
        var mlp_fc1 = new DenseLinear(n_embd, mlp_hidden);
        var mlp_activation = new Activation(ActivationFunctions.ReLU);

        // Output layer: mlp_hidden -> n_embd
        var mlp_fc2 = new DenseLinear(mlp_hidden, n_embd);

        // Normalization + MLP
        var mlp_normalized = new SequentialBlock(new INetworkModule[]
        {
            new LayerNorm(n_embd),
            mlp_fc1,
            mlp_activation,
            mlp_fc2
        });

        // Residual: MLP + skip
        var mlp_residual = new ResidualAdd(mlp_normalized, new Identity());

        // Complete transformer layer
        var transformer_layer = new SequentialBlock(new INetworkModule[]
        {
            attention_residual,
            mlp_residual
        });

        // ===== Output Head =====
        // Final normalization
        var final_norm = new LayerNorm(n_embd);

        // Project to vocabulary logits: n_embd -> vocab_size
        var output_head = new DenseLinear(n_embd, vocab_size);

        // ===== Complete Architecture =====
        var complete_model = new SequentialBlock(new INetworkModule[]
        {
            embedding_combine,      // Combines token + position embeddings
            transformer_layer,      // Single transformer block
            final_norm,            // Final layer normalization
            output_head            // Output projection to vocab
        });

        return new ArchitectureBlock(
            name: "MiniGpt",
            inputShape: new Shape(1, block_size), // batch, block_size (token indices)
            rootModule: complete_model
        );
    }
}