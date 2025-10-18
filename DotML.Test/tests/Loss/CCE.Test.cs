using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Loss;

[TestClass]
public class CategoricalCrossEntropyTest
{
    [TestMethod]
    public void TestInvoke()
    {
        var cce = new CategoricalCrossEntropy();

        float[] logits = [0.8448f, -0.2264f, 1.0283f, -0.4268f, 0.6123f];
        float[] label = [0f, 0f, 0f, 0f, 1f];
        float loss = 1.5181260108947754f;

        Assert.AreEqual(loss, cce.Invoke(logits.AsSpan(), label.AsSpan()), 0.0001f);

    }
    
    [TestMethod]
    public void TestGradient()
    {
        var cce = new CategoricalCrossEntropy();
        
        float[] logits = [0.8448f, -0.2264f, 1.0283f, -0.4268f, 0.6123f];
        float[] label = [0f, 0f, 0f, 0f, 1f];
        float[] grad = [0.2765f, 0.0947f, 0.3322f, 0.0775f, -0.7809f];

        float[] computed = new float[grad.Length];
        cce.Gradient(computed, logits, label);
        for (var i = 0; i < computed.Length; i++)
        {
            Assert.AreEqual(grad[i], computed[i], 0.0001f);
        }
    }
}