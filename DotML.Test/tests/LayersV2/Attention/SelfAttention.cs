using System.Text.Json;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Test.Layers.Attention;

[TestClass]
public class SelfAttentionTest {

    class SelfAttentionWeights
    {
        public float[][][]? X {get; set;}
        public float[][][]? Q_base {get; set;}
        public float[][][]? K_base {get; set;}
        public float[][][]? V_base {get; set;}
        public float[][][][]? Q {get; set;}
        public float[][][][]? K {get; set;}
        public float[][][][]? V {get; set;}
        public float[][][][]? scores {get; set;}
        public float[][][][]? attn_weights {get; set;}
        public float[][][]? attn_output_per_head {get; set;}
        public float[][][]? Y {get; set;}
        public float[][][]? dY {get; set;}
        public float[][][]? dX {get; set;}
        public float[][]? Q_proj_W {get; set;}
        public float[]? Q_proj_b {get; set;}
        public float[][]? K_proj_W {get; set;}
        public float[]? K_proj_b {get; set;}
        public float[][]? V_proj_W {get; set;}
        public float[]? V_proj_b {get; set;}
        public float[][]? dQ_proj_W {get; set;}
        public float[]? dQ_proj_b {get; set;}
        public float[][]? dK_proj_W {get; set;}
        public float[]? dK_proj_b {get; set;}
        public float[][]? dV_proj_W {get; set;}
        public float[]? dV_proj_b {get; set;}
        public float[][]? Output_proj_W {get; set;}
        public float[]? Output_proj_b {get; set;}
        public float[][]? dOutput_proj_W {get; set;}
        public float[]? dOutput_proj_b {get; set;}
    }

    private static void RestoreWeights(SelfAttention layer, SelfAttentionWeights weights)
    {
        // Restores the layer to the state it was initialized to for the forward pass of the test

        layer.QueryProjection.Weights = Tensor<float>.FromJaggedArray(weights.Q_proj_W!);
        layer.KeyProjection.Weights = Tensor<float>.FromJaggedArray(weights.K_proj_W!);
        layer.ValueProjection.Weights = Tensor<float>.FromJaggedArray(weights.V_proj_W!);
        
        layer.QueryProjection.Biases = Tensor<float>.FromJaggedArray(weights.Q_proj_b!);
        layer.KeyProjection.Biases = Tensor<float>.FromJaggedArray(weights.K_proj_b!);
        layer.ValueProjection.Biases = Tensor<float>.FromJaggedArray(weights.V_proj_b!);

        layer.OutputProjection.Weights = Tensor<float>.FromJaggedArray(weights.Output_proj_W!);
        layer.OutputProjection.Biases = Tensor<float>.FromJaggedArray(weights.Output_proj_b!);
    }

    private static void Test(int d_model, int d_k, int heads)
    {
        
        var layer = new SelfAttention(d_model: d_model, d_k: d_k, heads: heads);
        var weights = JsonSerializer.Deserialize<SelfAttentionWeights>(ResourceLoader.Find($"selfattention.{d_model}-{d_k}-{heads}.tensors"));
        if (weights is null)
            throw new Exception("Failed to load weights");
        RestoreWeights(layer, weights);

        var true_input = Tensor<float>.FromJaggedArray(weights.X!);
        var true_output = Tensor<float>.FromJaggedArray(weights.Y!);

        EvaluationContext ctx = new EvaluationContext(EvaluationMode.Training); // Training mode can trigger different behaviors that inference mode (not useful in this case though)
        var pred_output = layer.Forward(true_input, ctx);
        Assert.AreEqual(true, true_output.Equals(pred_output, delta: 0.001f));

        var true_dY = Tensor<float>.FromJaggedArray(weights.dY!);
        var true_dX = Tensor<float>.FromJaggedArray(weights.dX!);
        var pred_dX = layer.Backward(true_dY, ctx);
        Assert.AreEqual(true, true_dX.Equals(pred_dX.dX, delta: 0.001f));
    }

    [TestMethod]
    public void Test441()
    {
        const int d_model = 4;
        const int d_k = 4;
        const int heads = 1;

        Test(d_model, d_k, heads);
    }
}