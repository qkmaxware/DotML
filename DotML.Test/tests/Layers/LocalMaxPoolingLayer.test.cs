using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Layers;

[TestClass]
public class LocalMaxPoolingLayerTest {
    [TestMethod]
    public void TestSafetensors() {
        var layer = new LocalMaxPoolingLayer(new Shape3D(1, 4, 4), 2, 2);
        var writer = new LayerSafetensorWriter(); 
        layer.Visit(writer, 0);
        var tensors = writer.ToSafetensors();
        Assert.AreEqual(0, tensors.Keys().Count(), "Layer should not have any tensors.");
    }

    [TestMethod]
    public void TestLocalMaxPooling() {
        var layer = new LocalMaxPoolingLayer(new Shape3D(1, 4, 4), 2, 2);
        Matrix<float> input = new Matrix<float>(new float[,] {
            {12, 20, 30, 00},
            {08, 12, 02, 00},
            {34, 70, 37, 04},
            {112, 100, 25, 12}
        });

        var outputs = layer.EvaluateSync(new FeatureSet<float>(input));
        Assert.AreEqual(1, outputs.Channels);
        var output = outputs[0];

        Matrix<double> result = new Matrix<double>(new double[,] {
            {20, 30},
            {112, 37}
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
    public void TestStride1Padding0Kernel3() {
        var layer = new LocalMaxPoolingLayer(new Shape3D(1, 5, 5), size: 3, stride: 1);

        var X = Matrix<double>.FromFlattened(5, 5, [
            0.45132818818092346,
            -1.5761557817459106,
            -0.6721860766410828,
            -2.0042014122009277,
            -1.0609326362609863,
            -1.678432822227478,
            0.056944191455841064,
            1.0239132642745972,
            1.9709233045578003,
            0.7276046276092529,
            1.160701870918274,
            0.03253338485956192,
            -0.3749595284461975,
            -0.6730366349220276,
            -0.07208921015262604,
            1.8418018817901611,
            -1.1823792457580566,
            1.186455488204956,
            -0.880447268486023,
            -0.631264865398407,
            -1.4471193552017212,
            1.9196826219558716,
            -0.6389657855033875,
            0.6240558624267578,
            1.5673837661743164
        ]).ToFloatSet();

        var Y_truth = Matrix<double>.FromFlattened(3, 3, [
            1.160701870918274,
            1.9709233045578003,
            1.9709233045578003,
            1.8418018817901611,
            1.9709233045578003,
            1.9709233045578003,
            1.9196826219558716,
            1.9196826219558716,
            1.5673837661743164
        ]).ToFloatSet();
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<float>(new FeatureSet<float>(X)))[0,0];
        Assert.AreEqual(Y_truth.Rows, Y_projected.Rows);
        Assert.AreEqual(Y_truth.Columns, Y_projected.Columns);

        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        var dY = Matrix<double>.FromFlattened(3, 3, [
            -0.31870266795158386,
            -0.391875684261322,
            0.10528147220611572,
            -0.46503812074661255,
            1.715201735496521,
            -1.2883495092391968,
            2.3749876022338867,
            -0.5080739259719849,
            1.6750047206878662
        ]).ToFloatSet();
        var dX_truth = Matrix<double>.FromFlattened(5, 5, [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.14025795459747314,
            0.0,
            -0.31870266795158386,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.46503812074661255,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.8669136762619019,
            0.0,
            0.0,
            1.6750047206878662
        ]);

        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<float>(new FeatureSet<float>(X)),
            output: new BatchedFeatureSet<float>(new FeatureSet<float>(Y_truth)),
            error: new BatchedFeatureSet<float>(new FeatureSet<float>(dY))
        ));
        var dX_projected = backprop_returns.dX[0,0];
        Assert.IsNull(backprop_returns.Gradients);
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        Assert.AreEqual(dX_truth.Shape, dX_projected.Shape);
        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }

    [TestMethod]
    public void TestStride1Padding1Kernel3() {
        var layer = new LocalMaxPoolingLayer(new Shape3D(1, 5, 5), size: 3, stride: 1, padding: 1);

        var X_true = BatchedFeatureSet<double>.FromJagged(
            [
                [
                    [
                        [
                            -0.5640289783477783,
                            0.8789940476417542,
                            -0.034688565880060196,
                            0.34416288137435913,
                            0.8007024526596069
                        ],
                        [
                            -0.26175856590270996,
                            -0.6754032969474792,
                            0.0012084355112165213,
                            0.4772392809391022,
                            1.313050389289856
                        ],
                        [
                            0.38987675309181213,
                            2.471937894821167,
                            -0.7539513111114502,
                            1.551468014717102,
                            0.22664353251457214
                        ],
                        [
                            -0.23519648611545563,
                            -0.4340766966342926,
                            0.8029924631118774,
                            -0.9479889869689941,
                            -0.4330703914165497
                        ],
                        [
                            0.7628260850906372,
                            0.10244156420230865,
                            0.8272832036018372,
                            -0.854284405708313,
                            -0.17836932837963104
                        ]
                    ]
                ]
            ]
        ).ToFloatSet();
        var Y_true = BatchedFeatureSet<double>.FromJagged(
            [
                [
                    [
                        [
                            0.8789940476417542,
                            0.8789940476417542,
                            0.8789940476417542,
                            1.313050389289856,
                            1.313050389289856
                        ],
                        [
                            2.471937894821167,
                            2.471937894821167,
                            2.471937894821167,
                            1.551468014717102,
                            1.551468014717102
                        ],
                        [
                            2.471937894821167,
                            2.471937894821167,
                            2.471937894821167,
                            1.551468014717102,
                            1.551468014717102
                        ],
                        [
                            2.471937894821167,
                            2.471937894821167,
                            2.471937894821167,
                            1.551468014717102,
                            1.551468014717102
                        ],
                        [
                            0.7628260850906372,
                            0.8272832036018372,
                            0.8272832036018372,
                            0.8272832036018372,
                            -0.17836932837963104
                        ]
                    ]
                ]
            ]
        ).ToFloatSet();
        var dY_true = BatchedFeatureSet<double>.FromJagged(
            [
                [
                    [
                        [
                            -0.8070335388183594,
                            0.7404472231864929,
                            0.11264840513467789,
                            0.02506282925605774,
                            -0.3693905770778656
                        ],
                        [
                            -0.7195416688919067,
                            -1.0002985000610352,
                            -0.4659970998764038,
                            -1.1590412855148315,
                            -0.24849741160869598
                        ],
                        [
                            -1.0208487510681152,
                            -0.8971890807151794,
                            -0.757496178150177,
                            0.6741929054260254,
                            -0.8940054774284363
                        ],
                        [
                            -0.5281499624252319,
                            -0.056599754840135574,
                            0.4583166837692261,
                            1.228080153465271,
                            0.9006924033164978
                        ],
                        [
                            -0.6183019280433655,
                            -0.7174305319786072,
                            -0.7131734490394592,
                            -0.5339505672454834,
                            -0.7032472491264343
                        ]
                    ]
                ]
            ]
        ).ToFloatSet();
        var dX_true = BatchedFeatureSet<double>.FromJagged(
            [
                [
                    [
                        [
                            0.0,
                            0.04606208950281143,
                            0.0,
                            0.0,
                            0.0
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            -0.34432774782180786
                        ],
                        [
                            0.0,
                            -4.987804412841797,
                            0.0,
                            0.5014212727546692,
                            0.0
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0
                        ],
                        [
                            -0.6183019280433655,
                            0.0,
                            -1.9645545482635498,
                            0.0,
                            -0.7032472491264343
                        ]
                    ]
                ]
            ]
        ).ToFloatSet();

        var Y_pred = layer.EvaluateSync(X_true);
        AssertExt.AreEqual(Y_true, Y_pred);

        var dX_pred = layer.Backpropagate(new BackpropagationArgs(
            layer: -1,
            input: X_true,
            output: Y_true,
            error: dY_true
        )).dX;
        //throw new Exception(dX_pred.ToJaggedArrayString());
        AssertExt.AreEqual(dX_true, dX_pred);
    }

    [TestMethod]
    public void TestStride2Padding0Kernel3() {
        var layer = new LocalMaxPoolingLayer(new Shape3D(1, 5, 5), size: 3, stride: 2);

        var X = Matrix<double>.FromFlattened(5, 5, [
                    -0.9107525944709778,
                    -0.1645437330007553,
                    1.2178658246994019,
                    0.3981785774230957,
                    1.174186110496521,
                    0.49316108226776123,
                    0.9180596470832825,
                    -0.22267141938209534,
                    0.5467230677604675,
                    -0.41014403104782104,
                    -0.6195800304412842,
                    -1.1108744144439697,
                    -0.15905028581619263,
                    -0.7108968496322632,
                    0.4088130295276642,
                    -0.5547062158584595,
                    -0.4025643467903137,
                    0.41047704219818115,
                    -1.0392210483551025,
                    0.783096969127655,
                    0.09192565828561783,
                    -0.31207722425460815,
                    0.5189661979675293,
                    1.3695857524871826,
                    0.9875826239585876
        ]).ToFloatSet();

        var Y_truth = Matrix<double>.FromFlattened(2, 2, [
                    1.2178658246994019,
                    1.2178658246994019,
                    0.5189661979675293,
                    1.3695857524871826
        ]).ToFloatSet();
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<float>(new FeatureSet<float>(X)))[0,0];
        Assert.AreEqual(Y_truth.Rows, Y_projected.Rows);
        Assert.AreEqual(Y_truth.Columns, Y_projected.Columns);

        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        var dY = Matrix<double>.FromFlattened(2, 2, [
                    0.19057179987430573,
                    -0.1136438176035881,
                    -0.4186047911643982,
                    -1.3989176750183105
        ]).ToFloatSet();
        var dX_truth = Matrix<double>.FromFlattened(5, 5, [
                    0.0,
                    0.0,
                    0.07692798227071762,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.4186047911643982,
                    -1.3989176750183105,
                    0.0
        ]);

        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<float>(new FeatureSet<float>(X)),
            output: new BatchedFeatureSet<float>(new FeatureSet<float>(Y_truth)),
            error: new BatchedFeatureSet<float>(new FeatureSet<float>(dY))
        ));
        var dX_projected = backprop_returns.dX[0,0];
        Assert.IsNull(backprop_returns.Gradients);
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("LocalMaxPoolingLayerTest.TestStride1Padding0Kernel3.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        Assert.AreEqual(dX_truth.Shape, dX_projected.Shape);
        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }
}