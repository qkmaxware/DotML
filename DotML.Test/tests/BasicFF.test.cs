using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test;

[TestClass]
public class BasicFFTest {
    [TestMethod]
    public void TestNOT() {
        var network = new FeedforwardNetwork(
            new DenseLinearLayer(1, 1) {
                Weights = new Matrix<float>(1, 1, -1.0f),
                Biases = new Vec<float>(1, 1.0f)
            }
        );
        var outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            1.0f
        }));
        Assert.AreEqual(false, outputs[0] > 0.5);
        outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            0.0f,
        }));
        Assert.AreEqual(true, outputs[0] > 0.5);
    }

    [TestMethod]
    public void TestAND() {
        var network = new FeedforwardNetwork(
            new DenseLinearLayer(2, 1) {
                Weights = Matrix<float>.FromJagged(new float[][]{ new float[]{ 0.4f, 0.4f } }),
            }
        );
        var outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            1.0f,
            1.0f
        }));
        Assert.AreEqual(true, outputs[0] > 0.5);
        outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            1.0f,
            0.0f
        }));
        Assert.AreEqual(false, outputs[0] > 0.5);
        outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            0.0f,
            1.0f
        }));
        Assert.AreEqual(false, outputs[0] > 0.5);
        outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            0.0f,
            0.0f
        }));
        Assert.AreEqual(false, outputs[0] > 0.5);
    }

    [TestMethod]
    public void TestOR() {
        var network = new FeedforwardNetwork(
            new DenseLinearLayer(2, 1) {
                Weights = Matrix<float>.FromJagged(new float[][]{ new float[]{ 0.6f, 0.6f } }),
            }
        );
        var outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            1.0f,
            1.0f
        }));
        Assert.AreEqual(true, outputs[0] > 0.5);
        outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            1.0f,
            0.0f
        }));
        Assert.AreEqual(true, outputs[0] > 0.5);
        outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            0.0f,
            1.0f
        }));
        Assert.AreEqual(true, outputs[0] > 0.5);
        outputs = network.PredictSync(Vec<float>.Wrap(new float[]{
            0.0f,
            0.0f
        }));
        Assert.AreEqual(false, outputs[0] > 0.5);
    }

    [TestMethod]
    public void TestXOR() {
        var test_inputs = new float[][] {
            [1.0f, 1.0f],
            [1.0f, 0.0f],
            [0.0f, 1.0f],
            [0.0f, 0.0f]
        };
        var expected_outputs = new bool[]{
            false,
            true,
            true,
            false
        };

        // Weights/Biases copied from video: https://www.youtube.com/watch?v=kNPGXgzxoHw
        var network = new FeedforwardNetwork(
            new DenseLinearLayer(2, 2) {
                Weights = Matrix<float>.FromJagged([[20, 20], [-20, -20]]),
                Biases = Vec<float>.Wrap([-10, 30])
            },
            new ActivationLayer(new Shape3D(1, 2, 1), Sigmoid.Instance),
            new DenseLinearLayer(2, 1) {
                Weights = Matrix<float>.FromJagged([[20, 20]]),
                Biases = Vec<float>.Wrap([-30])
            },
            new ActivationLayer(new Shape3D(1, 1, 1), Sigmoid.Instance)
        );

        var actual_outputs = new float[test_inputs.Length];
        for (var i = 0; i < test_inputs.Length; i++) {
            actual_outputs[i] = network.PredictSync(Vec<float>.Wrap(test_inputs[i]))[0];
        }

        for (var i = 0; i < test_inputs.Length; i++) {
            Console.WriteLine("For input [" + string.Join(',', test_inputs[i]) + "]; expected [" + (expected_outputs[i] ? 1.0 : 0.0) + "] got output [" + actual_outputs[i] + "]");
        }

        for (var i = 0; i < test_inputs.Length; i++) {
            Assert.AreEqual(expected_outputs[i], actual_outputs[i] > 0.5, "Incorrect answer for input [" + string.Join(',', test_inputs[i]) + "]");
        }
    }

    [TestMethod]
    public void TestIteratingTrainingData() {
        var trainingData = new TrainingSet<double>(new List<TrainingPair<double>> {
            new TrainingPair<double> { Input = new Vec<double>(1.0, 1.0), Output = new Vec<double>(0.0) },
            new TrainingPair<double> { Input = new Vec<double>(1.0, 0.0), Output = new Vec<double>(1.0) },
            new TrainingPair<double> { Input = new Vec<double>(0.0, 1.0), Output = new Vec<double>(1.0) },
            new TrainingPair<double> { Input = new Vec<double>(0.0, 0.0), Output = new Vec<double>(0.0) }
        });

        var sequential = trainingData.SampleSequentially();
        var index = 0;
        while (sequential.MoveNext()) {
            Assert.AreEqual(trainingData[index], sequential.Current);
            index ++; 
        }
        Assert.AreEqual(trainingData.Size, index);

        var random = trainingData.SampleRandomly();
        index = 0;
        var set = new Dictionary<TrainingPair<double>, int>();
        foreach (var pair in trainingData) {
            set[pair] = 0;
        }
        while (random.MoveNext()) {
            if (set.ContainsKey(random.Current)) {
                set[random.Current] += 1;
            } else {
                set[random.Current] = 1;
            }
            index++;
        }
        Assert.AreEqual(false, set.Where(x => x.Value > 1).Any(), "Some training pairs were used more than once");
        Assert.AreEqual(false, set.Where(x => x.Value < 1).Any(), "Some training pairs were not used at all");
        Assert.AreEqual(trainingData.Size, index);
    }
}