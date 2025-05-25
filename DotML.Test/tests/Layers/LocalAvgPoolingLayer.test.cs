using DotML.Network;
using DotML.Network.Training;

namespace DotML.Test.Layers;

[TestClass]
public class LocalAvgPoolingLayerTest {
    [TestMethod]
    public void TestSafetensors() {
        var layer = new LocalAvgPoolingLayer(new Shape3D(1, 4, 4), 2, 2);
        var writer = new LayerSafetensorWriter(); 
        layer.Visit(writer, 0);
        var tensors = writer.ToSafetensors();
        Assert.AreEqual(0, tensors.Keys().Count(), "Layer should not have any tensors.");
    }

    [TestMethod]
    public void TestAvgMaxPooling() {
        var layer = new LocalAvgPoolingLayer(new Shape3D(1, 4, 4), 2, 2);
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
            {13, 8},
            {79, 19.5}
        });
        Assert.AreEqual(result.Rows, output.Rows);
        Assert.AreEqual(result.Columns, output.Columns);
        for (var r = 0; r < result.Rows; r++) {
            for (var c = 0; c < result.Columns; c++) {
                Assert.AreEqual(result[r, c], output[r, c], 0.01, $"Element mismatch @ row {r}, column {c}. Expected {result}, got {output}");
            }
        }
    }

    [TestMethod]
    public void TestStride1Padding0Kernel3() {
        var layer = new LocalAvgPoolingLayer(new Shape3D(1, 5, 5), size: 3, stride: 1);

        var X = Matrix<double>.FromFlattened(5, 5, [
            -0.06675183027982712,
            0.7418843507766724,
            -0.07433906942605972,
            -0.03677411377429962,
            -1.0757157802581787,

            0.7527914643287659,
            -0.2763320207595825,
            1.245193362236023,
            0.9814322590827942,
            0.2918619215488434,

            0.9653529524803162,
            -1.670727252960205,
            0.5376826524734497,
            -0.33155393600463867,
            -0.3139996826648712,

            0.37308916449546814,
            0.42045533657073975,
            0.4851781725883484,
            0.36806729435920715,
            -0.025358738377690315,

            0.4659097194671631,
            1.7895983457565308,
            0.6633204221725464,
            -0.22848817706108093,
            0.6940374970436096
        ]).ToFloatSet();

        var Y_truth = Matrix<double>.FromFlattened(3, 3, [
            0.2394171804189682,
            0.12405179440975189,
            0.1359763890504837,
            0.31474265456199646,
            0.19548842310905457,
            0.3598337471485138,
            0.4477621614933014,
            0.22594809532165527,
            0.20543172955513
        ]).ToFloatSet();
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<float>(new FeatureSet<float>(X)))[0,0];
        Assert.AreEqual(Y_truth.Rows, Y_projected.Rows);
        Assert.AreEqual(Y_truth.Columns, Y_projected.Columns);

        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        var dY = Matrix<double>.FromFlattened(3, 3, [
            -2.179100513458252,
            -0.08496836572885513,
            -1.8678022623062134,
            1.0132676362991333,
            -0.5630502700805664,
            -0.2114965319633484,
            -1.1144579648971558,
            -1.0145455598831177,
            -0.3372708261013031
        ]).ToFloatSet();
        var dX_truth = Matrix<double>.FromFlattened(5, 5, [
            -0.24212227761745453,
            -0.2515632212162018,
            -0.4590967893600464,
            -0.21697451174259186,
            -0.2075335830450058,
            -0.12953698635101318,
            -0.2015390694141388,
            -0.43257224559783936,
            -0.30303525924682617,
            -0.23103320598602295,
            -0.2533656358718872,
            -0.4380950331687927,
            -0.7066026926040649,
            -0.4532370865345001,
            -0.2685077488422394,
            -0.011243373155593872,
            -0.18653179705142975,
            -0.24750596284866333,
            -0.23626258969306946,
            -0.06097415089607239,
            -0.12382866442203522,
            -0.23655594885349274,
            -0.274030476808548,
            -0.15020182728767395,
            -0.037474535405635834
        ]).ToFloatSet();

        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<float>(new FeatureSet<float>(X)),
            output: new BatchedFeatureSet<float>(new FeatureSet<float>(Y_truth)),
            error: new BatchedFeatureSet<float>(new FeatureSet<float>(dY))
        ));
        var dX_projected = backprop_returns.dX[0,0];
        Assert.IsNull(backprop_returns.Gradients);
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        Assert.AreEqual(dX_truth.Shape, dX_projected.Shape);
        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }

    [TestMethod]
    public void TestStride1Padding1Kernel3() {
        var layer = new LocalAvgPoolingLayer(new Shape3D(1, 5, 5), size: 3, stride: 1, padding: 1);

        var X_true = BatchedFeatureSet<double>.FromJagged(
            [
                [
                    [
                        [
                            0.45152169466018677,
                            0.8342990279197693,
                            -1.0050057172775269,
                            -1.827385663986206,
                            0.5015586614608765
                        ],
                        [
                            0.5459192395210266,
                            0.07884866744279861,
                            1.2718443870544434,
                            0.4100680947303772,
                            1.405443787574768
                        ],
                        [
                            0.14313803613185883,
                            0.6634364724159241,
                            1.1774097681045532,
                            0.11305268108844757,
                            -0.12511587142944336
                        ],
                        [
                            -0.6129352450370789,
                            -0.19878138601779938,
                            1.196243166923523,
                            0.608299732208252,
                            -1.5767254829406738
                        ],
                        [
                            -1.019062876701355,
                            0.3999039828777313,
                            1.7012710571289062,
                            0.9543417096138,
                            -0.3465256690979004
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
                            0.21228761970996857,
                            0.24193637073040009,
                            -0.026370134204626083,
                            0.08405819535255432,
                            0.05440942943096161
                        ],
                        [
                            0.3019070029258728,
                            0.4623790383338928,
                            0.1907297521829605,
                            0.21354113519191742,
                            0.05306907370686531
                        ],
                        [
                            0.06884730607271194,
                            0.47390255331993103,
                            0.5911579132080078,
                            0.49783557653427124,
                            0.09278032183647156
                        ],
                        [
                            -0.06936677545309067,
                            0.383402556180954,
                            0.7350196838378906,
                            0.4113612771034241,
                            -0.041408102959394455
                        ],
                        [
                            -0.1589861661195755,
                            0.16295985877513885,
                            0.5179197788238525,
                            0.2818782925605774,
                            -0.04006774723529816
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
                            0.6141520142555237,
                            -0.2858428359031677,
                            0.389507532119751,
                            -1.2025781869888306,
                            -0.4804628789424896
                        ],
                        [
                            -0.85873943567276,
                            0.7361997961997986,
                            0.7948909997940063,
                            0.2554594874382019,
                            0.3261907696723938
                        ],
                        [
                            0.242221400141716,
                            2.267869472503662,
                            0.6537995934486389,
                            0.013349994085729122,
                            -1.2659605741500854
                        ],
                        [
                            -1.8502533435821533,
                            2.2527825832366943,
                            -1.5930553674697876,
                            -0.8987794518470764,
                            -0.860501229763031
                        ],
                        [
                            -0.9780293703079224,
                            -0.17550688982009888,
                            0.3657093048095703,
                            -0.3458174169063568,
                            -0.23915183544158936
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
                            0.022863280028104782,
                            0.15446311235427856,
                            0.07640408724546432,
                            0.009223084896802902,
                            -0.12237675487995148
                        ],
                        [
                            0.30176225304603577,
                            0.5060064792633057,
                            0.40251731872558594,
                            -0.05731146037578583,
                            -0.2615557014942169
                        ],
                        [
                            0.3100089430809021,
                            0.29396840929985046,
                            0.49805745482444763,
                            -0.28606730699539185,
                            -0.2700267732143402
                        ],
                        [
                            0.1954537332057953,
                            0.13172635436058044,
                            0.28226128220558167,
                            -0.46337857842445374,
                            -0.3996511697769165
                        ],
                        [
                            -0.0834452360868454,
                            -0.21981701254844666,
                            -0.04385192692279816,
                            -0.3968440294265747,
                            -0.26047223806381226
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
        var layer = new LocalAvgPoolingLayer(new Shape3D(1, 5, 5), size: 3, stride: 2);

        var X = Matrix<double>.FromFlattened(5, 5, [
                    1.3925418853759766,
                    -0.06613739579916,
                    -0.5422175526618958,
                    0.5325701236724854,
                    -0.8462679982185364,
                    1.1723917722702026,
                    -0.9220021367073059,
                    -0.6057412624359131,
                    0.77365642786026,
                    1.1700763702392578,
                    0.2963123023509979,
                    -0.63601154088974,
                    1.5949783325195312,
                    0.1986580193042755,
                    -0.3485676348209381,
                    -0.06287429481744766,
                    0.7226407527923584,
                    1.2705717086791992,
                    0.8389192819595337,
                    -1.1125303506851196,
                    0.8723279237747192,
                    0.06361571699380875,
                    0.5719498991966248,
                    -0.9991300106048584,
                    -0.3620969355106354
        ]).ToFloatSet();

        var Y_truth = Matrix<double>.FromFlattened(2, 2, [
                    0.1871238350868225,
                    0.21412721276283264,
                    0.5215012431144714,
                    0.18363916873931885
        ]).ToFloatSet();
        var Y_projected = layer.EvaluateSync(new BatchedFeatureSet<float>(new FeatureSet<float>(X)))[0,0];
        Assert.AreEqual(Y_truth.Rows, Y_projected.Rows);
        Assert.AreEqual(Y_truth.Columns, Y_projected.Columns);

        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.YProjected.txt")) {
            writer.Write(Y_projected);
        }
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.YTruth.txt")) {
            writer.Write(Y_truth);
        }
        foreach (var (projected, truth) in Y_projected.Zip(Y_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Feedforward failed, projected value does not equal truth. Compare YProjected.txt to YTruth.txt.");
        }

        var dY = Matrix<double>.FromFlattened(2, 2, [
                    0.23149579763412476,
                    -0.8691568374633789,
                    -0.5454152226448059,
                    1.0452895164489746
        ]).ToFloatSet();
        var dX_truth = Matrix<double>.FromFlattened(5, 5, [
                    0.025721754878759384,
                    0.025721754878759384,
                    -0.07085122168064117,
                    -0.09657298028469086,
                    -0.09657298028469086,
                    0.025721754878759384,
                    0.025721754878759384,
                    -0.07085122168064117,
                    -0.09657298028469086,
                    -0.09657298028469086,
                    -0.03487993776798248,
                    -0.03487993776798248,
                    -0.015309639275074005,
                    0.019570298492908478,
                    0.019570298492908478,
                    -0.06060169264674187,
                    -0.06060169264674187,
                    0.05554158613085747,
                    0.11614327877759933,
                    0.11614327877759933,
                    -0.06060169264674187,
                    -0.06060169264674187,
                    0.05554158613085747,
                    0.11614327877759933,
                    0.11614327877759933
        ]).ToFloatSet();

        var backprop_returns = layer.Backpropagate(new BackpropagationArgs(
            layer: -1,
            input: new BatchedFeatureSet<float>(new FeatureSet<float>(X)),
            output: new BatchedFeatureSet<float>(new FeatureSet<float>(Y_truth)),
            error: new BatchedFeatureSet<float>(new FeatureSet<float>(dY))
        ));
        var dX_projected = backprop_returns.dX[0,0];
        Assert.IsNull(backprop_returns.Gradients);
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.dXProjected.txt")) {
            writer.Write(dX_projected);
        }
        using (var writer = new StreamWriter("LocalAvgPoolingLayerTest.TestStride1Padding0Kernel3.dXTruth.txt")) {
            writer.Write(dX_truth);
        }

        Assert.AreEqual(dX_truth.Shape, dX_projected.Shape);
        foreach (var (projected, truth) in dX_projected.Zip(dX_truth)) {
            Assert.AreEqual(truth, projected, 0.001, "Backprop failed, gradient(X) value does not equal truth. Compare dXProjected.txt to dXTruth.txt.");
        }
    }
}