using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using DotML;
using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Test.Layers.Residual;

[TestClass]
public class ResidualOutputTest
{
    [TestMethod]
    public void TestAdd()
    {
        // Translated from: DotML.Utils\BackpropGenerators\residual.py
        const int in_feats = 5;
        const int hidden_feats = 4;
        const int out_feats = 3;

        var data = JsonSerializer.Deserialize<TestAddData>(ResourceLoader.Find($"residual_add_{in_feats}{hidden_feats}{out_feats}.tensors.json"));
        if (AnyNull(data, out var nullish))
        {
            Assert.Fail($"Missing test data '{nullish}'.");
            return;
        }     

        #nullable disable
        var block = new ResidualAdd(
            new SequentialBlock([
                new DenseLinear(in_feats, hidden_feats) {
                    Weights = Tensor<float>.FromFlattenedArray(new Shape(hidden_feats, in_feats), data.main_W1),
                    Biases = Tensor<float>.Vec(data.main_b1),
                },
                new Network.Activation(ActivationFunctions.ReLU),
                new DenseLinear(hidden_feats, out_feats) {
                    Weights = Tensor<float>.FromFlattenedArray(new Shape(out_feats, hidden_feats), data.main_W2),
                    Biases = Tensor<float>.Vec(data.main_b2),
                }
            ]),
            (in_feats != out_feats) 
                ? (new DenseLinear(in_feats, out_feats)
                {
                    Weights = Tensor<float>.FromFlattenedArray(new Shape(out_feats, in_feats), data.skip_W),
                    Biases = Tensor<float>.Vec(data.skip_b),
                }) 
                : (new Identity())
        );

        
        const int batches = 1; 
        const float accuracy = 0.0001f;
        // Test forward propagation 
        var x_true = Tensor<float>.FromFlattenedArray(new Shape(batches, in_feats), data.X);
        var ctx = new EvaluationContext(EvaluationMode.Training);
        var y_pred = block.Forward(x_true, ctx);
        var y_true = Tensor<float>.FromFlattenedArray(new Shape(batches, out_feats), data.Y);
        Assert.IsTrue(y_pred.Equals(y_true, accuracy));
        
        // Could test the inner layers too but like why those should already be tested by their own unit tests so no real point unless debugging an issue with the block is hard

        // Test backward propagation
        var dy_true = Tensor<float>.FromFlattenedArray(new Shape(batches, out_feats), data.dY);
        var grads = block.Backward(dy_true, ctx);
        var dx_true = Tensor<float>.FromFlattenedArray(new Shape(batches, in_feats), data.dX);
        Assert.IsTrue(grads.dX.Equals(dx_true, accuracy));
        #nullable restore
    }

    private class TestAddData
    {
        public float[]? X {get; set;}
        public float[]? Y {get; set;}
        public float[]? dY {get; set;}

        public float[]? main_W1 {get; set;}
        public float[]? main_b1 {get; set;}
        public float[]? main_dW1 {get; set;}
        public float[]? main_db1 {get; set;}

        public float[]? main_W2 {get; set;}
        public float[]? main_b2 {get; set;}
        public float[]? main_dW2 {get; set;}
        public float[]? main_db2 {get; set;}

        public float[]? skip_W {get; set;}
        public float[]? skip_b {get; set;}
        public float[]? skip_dW {get; set;}
        public float[]? skip_db {get; set;}

        public float[]? dX {get; set;}
    }
    private static bool AnyNull(object? data, [NotNullWhen(true)]out string? nullish)
    {
        nullish = null;
        if (data is null) {
            nullish = nameof(data);
            return true;
        }

        var props = data.GetType().GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
        foreach (var prop in props)
        {
            if (!prop.CanRead)
                continue;

            var val = prop.GetValue(data);
            if (val is null) {
                nullish = prop.Name;
                return true;
            }
        }
        return false;
    }
}