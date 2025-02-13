using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Layers;

[TestClass]
public class ConvolutionalLayerTest {
    [TestMethod]
    public void TestConvolutionalLayerValidPadding() {
        var kernels = new Matrix<double>[]{
            new Matrix<double>(new double[,] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            })
        };
        var layer = new ConvolutionLayer(new Shape3D(1, 5, 5), Padding.Valid, new ConvolutionFilter(kernels));
        var input = new Matrix<double>[] {
            new Matrix<double>(new double[,]{
                {1, 1, 1, 0, 0},
                {0, 1, 1, 1, 0},
                {0, 0, 1, 1, 1},
                {0, 0, 1, 1, 0},
                {0, 1, 1, 0, 0},
            }),
        };
        var outputs = layer.EvaluateSync((FeatureSet<double>)input);
        Assert.AreEqual(1, outputs.Channels);
        var output = outputs[0];

        Matrix<double> result = new Matrix<double>(new double[,] {
            {4, 3, 4},
            {2, 4, 3},
            {2, 3, 4}
        });

        Assert.AreEqual(result.Rows, output.Rows);
        Assert.AreEqual(result.Columns, output.Columns);
        for (var r = 0; r < result.Rows; r++) {
            for (var c = 0; c < result.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestConvolutionalLayerSamePadding() {
        var kernels = new Matrix<double>[]{
            new Matrix<double>(new double[,] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            })
        };
        var layer = new ConvolutionLayer(new Shape3D(1, 5, 5), Padding.Same, new ConvolutionFilter(kernels));
        Matrix<double> input = new Matrix<double>(new double[,]{
            {1, 1, 1, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 1, 1, 1},
            {0, 0, 1, 1, 0},
            {0, 1, 1, 0, 0},
        });
        var inputs = new Matrix<double>[] {
            input
        };
        var outputs = layer.EvaluateSync((FeatureSet<double>)inputs);
        Assert.AreEqual(1, outputs.Channels);
        var output = outputs[0];

        // This is a correct answer for this particular problem (ie same padding, no stride)
        var fft_output = CooleyTukey.ConvolveFFT(inputs[0], kernels[0], layer.StrideX, layer.StrideY, layer.ColumnsPadding, layer.RowsPadding);

        Matrix<double> result = new Matrix<double>(new double[,]{
            {2, 2, 3, 1, 1},
            {1, 4, 3, 4, 1},
            {1, 2, 4, 3, 3},
            {1, 2, 3, 4, 1},
            {0, 2, 2, 1, 1},
        });

        Assert.AreEqual(input.Rows, output.Rows);
        Assert.AreEqual(input.Columns, output.Columns);

        for (var r = 0; r < output.Rows; r++) {
            for (var c = 0; c < output.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], 0.0001, $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
                Assert.AreEqual(result[r, c], fft_output[r, c], 0.0001, $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestStride1PaddingSame() {
        // SRC Generator = /DotML.Utils/BackpropGenerators/conv2d.py
        // Step 1: Setup layer and input
        var layer = new ConvolutionLayer(
            input_size: new Shape3D(1, 5, 5),
            padding: Padding.Same,
            stride: 1,
            new ConvolutionFilter(
                Matrix<double>.FromFlattened(3, 3, [
                    -0.33186790347099304,
                    -0.11537425220012665,
                    -0.20978042483329773,
                    0.08112597465515137,
                    -0.16276657581329346,
                    0.051479071378707886,
                    -0.3236038386821747,
                    -0.1491357535123825,
                    0.3179304301738739
                ])
            )
        );
        layer.Filters[0].Bias = 0.25102999806404114;

        var X = Matrix<double>.FromFlattened(5, 5, [
            1.6284466981887817,
            1.4562504291534424,
            0.5519229173660278,
            0.2747989594936371,
            -0.6401860117912292,
            1.720615267753601,
            1.4433258771896362,
            -0.6038796901702881,
            -0.2045498490333557,
            0.2927928566932678,
            -0.8961163759231567,
            -0.7354010343551636,
            0.9076979756355286,
            -0.9748172163963318,
            1.4172523021697998,
            -0.7628729939460754,
            0.8854146599769592,
            1.4145307540893555,
            0.3223850727081299,
            -0.6889523863792419,
            -0.5507891178131104,
            0.41970959305763245,
            -0.1417425125837326,
            -2.1431992053985596,
            -0.4211532473564148
        ]);

        // Step 2: Assert that feed-forward worked.
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<double>(new FeatureSet<double>(X)))[0, 0];
        var Y_truth = Matrix<double>.FromFlattened(5, 5, [
            0.2632116973400116,
            -0.7895180583000183,
            -0.14855682849884033,
            0.537132203578949,
            0.40005144476890564,
            -0.5482646226882935,
            -0.011375358328223228,
            -0.35604098439216614,
            0.4720677137374878,
            0.2735344469547272,
            0.253005713224411,
            0.29844436049461365,
            -0.7679511904716492,
            -0.005985418800264597,
            -0.02620915323495865,
            0.8940228819847107,
            0.3802448511123657,
            -0.34302252531051636,
            0.02336486242711544,
            1.3056749105453491,
            0.2645595669746399,
            -0.014986595138907433,
            -0.3268508017063141,
            0.20458804070949554,
            0.11820864677429199
        ]);

        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingSame.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingSame.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        // Step 3: Backpropagate and check gradients
        var dY = Matrix<double>.FromFlattened(5, 5, [
            -0.8490618467330933,
            1.3524293899536133,
            0.8703511953353882,
            1.7506407499313354,
            -1.1928188800811768,
            -0.9589080810546875,
            -2.0769541263580322,
            -1.091245412826538,
            0.9739490747451782,
            -0.05606662109494209,
            -1.6003074645996094,
            -1.182836651802063,
            -1.3598333597183228,
            0.1821463257074356,
            -1.4911508560180664,
            1.843986988067627,
            1.8884780406951904,
            0.031288716942071915,
            -0.44852301478385925,
            0.6544036865234375,
            -0.3919405937194824,
            -0.4980011582374573,
            -1.8042224645614624,
            0.5716819167137146,
            1.3351649045944214
        ]);
        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1, 
            input: new BatchedFeatureSet<double>(new FeatureSet<double>(X)),
            output: new BatchedFeatureSet<double>(new FeatureSet<double>(Y_truth)),
            error: new BatchedFeatureSet<double>(new FeatureSet<double>(dY))
        ));
        Assert.IsInstanceOfType<ConvolutionLayer.Gradients>(backprop_returns.Gradients);
        var gradients = (ConvolutionLayer.Gradients)backprop_returns.Gradients;
        
        var dX_projected = backprop_returns.InputErrors[0, 0];
        var dX_truth = Matrix<double>.FromFlattened(5, 5, [
            1.0478237867355347,
            0.6097053289413452,
            0.308363676071167,
            -0.2017495334148407,
            0.08642558753490448,
            0.253737211227417,
            0.3703465759754181,
            0.2279522567987442,
            0.9414942264556885,
            0.9275673627853394,
            0.14016170799732208,
            -0.25725847482681274,
            -0.8884612321853638,
            -0.8666599988937378,
            0.5886858701705933,
            0.6849891543388367,
            0.6361895799636841,
            -0.05355694890022278,
            0.020184457302093506,
            -0.12328216433525085,
            -0.8627291321754456,
            0.20900672674179077,
            1.0552908182144165,
            -0.21254292130470276,
            -0.42808467149734497
        ]);
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingSame.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingSame.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        var dB_projected = gradients.BiasGradients;
        var dB_truth = Vec<double>.Wrap([-3.547349214553833]);
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingSame.dBProjected.txt")) {
            writer.Write(dB_projected);
        }
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingSame.dBTruth.txt")) {
            writer.Write(dB_truth);
        }
        
        var dW_projected = gradients.FilterKernelGradients[0, 0];
        var dW_truth = Matrix<double>.FromFlattened(3, 3, [
                -10.99089527130127,
                -14.076019287109375,
                -7.06515645980835,
                -3.2001407146453857,
                -4.924467086791992,
                8.84919548034668,
                3.4817280769348145,
                -0.5803017616271973,
                -3.469475746154785
        ]);
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingSame.dWProjected.txt")) {
            writer.Write(dW_projected);
        }
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingSame.dWTruth.txt")) {
            writer.Write(dW_truth);
        }

        foreach (var (projected, truth) in dB_projected.Zip(dB_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(B) value does not equal truth. Compare dBProjected.txt to dBTruth.txt.");
        }
        foreach (var (projected, truth) in dW_projected.Zip(dW_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(W) value does not equal truth. Compare dWProjected.txt to dWTruth.txt.");
        }
        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }

    [TestMethod]
    public void TestStride1PaddingValid() {
        // SRC Generator = /DotML.Utils/BackpropGenerators/conv2d.py
        // Step 1: Setup layer and input
        var layer = new ConvolutionLayer(
            input_size: new Shape3D(1, 5, 5),
            padding: Padding.Valid,
            stride: 1,
            new ConvolutionFilter(
                Matrix<double>.FromFlattened(3, 3, [
                    0.1519714593887329,
                    -0.017021745443344116,
                    -0.17113947868347168,
                    0.3106711208820343,
                    -0.24575552344322205,
                    0.30953022837638855,
                    -0.23382768034934998,
                    -0.22341418266296387,
                    -0.255176305770874
                ])
            )
        );
        layer.Filters[0].Bias = -0.058894991874694824;

        var X = Matrix<double>.FromFlattened(5, 5, [
            1.1838496923446655,
            0.4641179144382477,
            -1.9392775297164917,
            0.27759337425231934,
            -0.07365334033966064,
            -2.390148162841797,
            1.0974578857421875,
            -0.3438238799571991,
            0.5217018127441406,
            0.5510475039482117,
            -0.05441654846072197,
            -1.506787896156311,
            0.4322599172592163,
            2.035956382751465,
            0.3526442050933838,
            0.08510291576385498,
            -1.2911185026168823,
            1.0010817050933838,
            -0.28336191177368164,
            -0.8107024431228638,
            -0.4596104919910431,
            -0.49612998962402344,
            0.10646743327379227,
            0.7952219843864441,
            -0.7875760197639465
        ]);

        // Step 2: Assert that feed-forward worked.
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<double>(new FeatureSet<double>(X)))[0, 0];
        var Y_truth = Matrix<double>.FromFlattened(3, 3, [
            -0.4346175491809845,
            0.3202958106994629,
            -1.0561137199401855,
            0.11832758784294128,
            0.2308509200811386,
            -0.4351370334625244,
            0.729254424571991,
            -1.4892170429229736,
            0.03990913927555084
        ]);

        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingValid.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingValid.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        Assert.AreEqual(Y_truth.Shape, Y_projected.Shape);
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        // Step 3: Backpropagate and check gradients
        var dY = Matrix<double>.FromFlattened(3, 3, [
            0.8627731204032898,
            -1.1305592060089111,
            -0.4596337676048279,
            -0.4029831886291504,
            -1.2035309076309204,
            -1.0401866436004639,
            -0.03843408823013306,
            1.1006325483322144,
            1.9555470943450928
        ]);
        Assert.AreEqual(Y_truth.Shape, dY.Shape);
        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1, 
            input: new BatchedFeatureSet<double>(new FeatureSet<double>(X)),
            output: new BatchedFeatureSet<double>(new FeatureSet<double>(Y_truth)),
            error: new BatchedFeatureSet<double>(new FeatureSet<double>(dY))
        ));
        Assert.IsInstanceOfType<ConvolutionLayer.Gradients>(backprop_returns.Gradients);
        var gradients = (ConvolutionLayer.Gradients)backprop_returns.Gradients;
        
        var dX_projected = backprop_returns.InputErrors[0, 0];
        var dX_truth = Matrix<double>.FromFlattened(5, 5, [
            0.13111689686775208,
            -0.18649862706661224,
            -0.19826167821884155,
            0.20130707323551178,
            0.0786614865064621,
            0.20679675042629242,
            -0.7393062114715576,
            0.33347442746162415,
            -0.013307243585586548,
            0.03574645519256592,
            -0.33277636766433716,
            -0.035347700119018555,
            0.27281200885772705,
            0.05263453722000122,
            -0.5393528938293457,
            0.08228826522827148,
            0.722831130027771,
            0.940091073513031,
            0.3995975852012634,
            0.8707319498062134,
            0.008986953645944595,
            -0.24877162277698517,
            -0.6933504939079285,
            -0.7177523374557495,
            -0.49900928139686584
        ]);
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingValid.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingValid.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        var dB_projected = gradients.BiasGradients;
        var dB_truth = Vec<double>.Wrap([-0.3563750982284546]);
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingValid.dBProjected.txt")) {
            writer.Write(dB_projected);
        }
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingValid.dBTruth.txt")) {
            writer.Write(dB_truth);
        }
        
        var dW_projected = gradients.FilterKernelGradients[0, 0];
        var dW_truth = Matrix<double>.FromFlattened(3, 3, [
            0.577020525932312,
            6.409263610839844,
            -0.10182112455368042,
            -1.225755214691162,
            -0.3377056121826172,
            -6.066802024841309,
            1.615986704826355,
            -1.4229464530944824,
            -1.9790031909942627
        ]);
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingValid.dWProjected.txt")) {
            writer.Write(dW_projected);
        }
        using (var writer = new StreamWriter("ConvolutionalLayerTest.TestStride1PaddingValid.dWTruth.txt")) {
            writer.Write(dW_truth);
        }

        Assert.AreEqual(dB_truth.Dimensionality, dB_projected.Dimensionality);
        foreach (var (projected, truth) in dB_projected.Zip(dB_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(B) value does not equal truth. Compare dBProjected.txt to dBTruth.txt.");
        }
        Assert.AreEqual(dW_truth.Shape, dW_projected.Shape);
        foreach (var (projected, truth) in dW_projected.Zip(dW_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(W) value does not equal truth. Compare dWProjected.txt to dWTruth.txt.");
        }
        Assert.AreEqual(dX_truth.Shape, dX_projected.Shape);
        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }
}