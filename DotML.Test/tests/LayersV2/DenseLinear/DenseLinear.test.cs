using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;
using Microsoft.VisualBasic;

namespace DotML.Test.Layers.Dense;

[TestClass]
public class DenseLinearTest
{
    private static void Test(DenseLinear layer, Tensor<float> w, Tensor<float> b, Tensor<float> x, Tensor<float> y, Tensor<float> dy, Tensor<float> dx, Tensor<float> dw, Tensor<float> db)
    {
        // Init
        Assert.AreEqual(layer.Weights.Shape, w.Shape);
        layer.Weights = w;
        Assert.AreEqual(layer.Biases.Shape, b.Shape);
        layer.Biases = b;

        // Forward
        var context = new EvaluationContext();
        var y_projected = layer.Forward(x, context);
        Assert.AreEqual(true, y_projected.Equals(y, 0.0001f));

        // Backwards
        var back = layer.Backward(dy, context);
        Assert.IsInstanceOfType<WeightAndBiasGradients>(back);
        WeightAndBiasGradients gradients = (WeightAndBiasGradients)back;

        Assert.AreEqual(true, gradients.dB.Equals(db, 0.001f));
        Assert.AreEqual(true, gradients.dW.Equals(dw, 0.001f));
        Assert.AreEqual(true, gradients.dX.Equals(dx, 0.001f));
    }

    [TestMethod]
    public void TestBatch1Channels1In5Out3Flat()
    {
        double[][][][] x = [
            [
                [
                    [
                        -0.6900870203971863
                    ],
                    [
                        0.7155113220214844
                    ],
                    [
                        -0.9192900061607361
                    ],
                    [
                        -0.31789007782936096
                    ],
                    [
                        0.30499646067619324
                    ]
                ]
            ]
        ];

        double[][] w = [
            [
                -0.27194780111312866,
                -0.16708391904830933,
                -0.17664295434951782,
                -0.02015891671180725,
                -0.07219260931015015
            ],
            [
                -0.26477667689323425,
                0.049975425004959106,
                0.022173643112182617,
                0.2885233163833618,
                0.06693542003631592
            ],
            [
                -0.09493818879127502,
                0.010637015104293823,
                -0.007061272859573364,
                -0.3193921148777008,
                -0.26885488629341125
            ]
        ];

        double[] b = [
            -0.29440563917160034,
            -0.28308242559432983,
            -0.38278505206108093
        ];

        double[][][][] y = [
            [
                [
                    [
                        -0.07951249182224274
                    ],
                    [
                        -0.1562931388616562
                    ],
                    [
                        -0.28363537788391113
                    ]
                ]
            ]
        ];

        double[][][][] dy = [
            [
                [
                    [
                        0.5209593176841736
                    ],
                    [
                        1.0958826541900635
                    ],
                    [
                        0.07956593483686447
                    ]
                ]
            ]
        ];

        double[][][][] dx = [
            [
                [
                    [
                        -0.43939176201820374
                    ],
                    [
                        -0.031430378556251526
                    ],
                    [
                        -0.06828591227531433
                    ],
                    [
                        0.2802729904651642
                    ],
                    [
                        0.01435226108878851
                    ]
                ]
            ]
        ];

        double[][] dw = [
            [
                -0.3595072627067566,
                0.37275227904319763,
                -0.47891268134117126,
                -0.16560779511928558,
                0.1588907539844513
            ],
            [
                -0.7562543749809265,
                0.7841164469718933,
                -1.0074340105056763,
                -0.348370224237442,
                0.3342403173446655
            ],
            [
                -0.0549074187874794,
                0.05693032592535019,
                -0.07314416766166687,
                -0.025293221697211266,
                0.024267328903079033
            ]
        ];

        double[] db = [
            0.5209593176841736,
            1.0958826541900635,
            0.07956593483686447
        ];

        Test(
            layer: new DenseLinear(5, 3),

            w: Tensor<double>.FromJaggedArray(w).ToFloat(),
            dw: Tensor<double>.FromJaggedArray(dw).ToFloat(),
            b: Tensor<double>.FromJaggedArray(b).ReshapeShared(new TensorShape(b.Length, 1)).ToFloat(),
            db: Tensor<double>.FromJaggedArray(db).ReshapeShared(new TensorShape(db.Length, 1)).ToFloat(),

            x: Tensor<double>.FromJaggedArray(x).ToFloat(),
            dx: Tensor<double>.FromJaggedArray(dx).ToFloat(),

            y: Tensor<double>.FromJaggedArray(y).ReshapeShared(new TensorShape(1, 3)).ToFloat(),
            dy: Tensor<double>.FromJaggedArray(dy).ReshapeShared(new TensorShape(1, 3)).ToFloat()
        );
    }
    
    [TestMethod]
    public void TestBatch3Channels1In5Out3Flat()
    {
        double[][][][] x = [
            [
                [
                    [
                        -1.2257734537124634
                    ],
                    [
                        2.3945205211639404
                    ],
                    [
                        -0.47454550862312317
                    ],
                    [
                        -0.8302198648452759
                    ],
                    [
                        -1.9457446336746216
                    ]
                ]
            ],
            [
                [
                    [
                        -1.0720231533050537
                    ],
                    [
                        -2.090482234954834
                    ],
                    [
                        -0.7478539347648621
                    ],
                    [
                        0.8958786129951477
                    ],
                    [
                        0.12809151411056519
                    ]
                ]
            ],
            [
                [
                    [
                        2.8023812770843506
                    ],
                    [
                        0.08298379927873611
                    ],
                    [
                        0.3140013813972473
                    ],
                    [
                        0.765845775604248
                    ],
                    [
                        -1.259905457496643
                    ]
                ]
            ]
        ];

        double[][] w = [
            [
                0.03159457445144653,
                -0.036105722188949585,
                0.2734866142272949,
                -0.24084720015525818,
                0.2165004014968872
            ],
            [
                -0.07852727174758911,
                0.30910271406173706,
                -0.17269793152809143,
                0.01492336392402649,
                -0.058808594942092896
            ],
            [
                -0.2178659439086914,
                -0.43815529346466064,
                -0.23246511816978455,
                0.1885082721710205,
                0.1378399133682251
            ]
        ];

        double[] b = [
            -0.07771638035774231,
            -0.38214579224586487,
            -0.040356189012527466
        ];

        double[][][][] y = [
            [
                [
                    [
                        -0.5539802312850952
                    ],
                    [
                        0.6382535696029663
                    ],
                    [
                        -1.1368629932403564
                    ]
                ]
            ],
            [
                [
                    [
                        -0.4286741614341736
                    ],
                    [
                        -0.8091470003128052
                    ],
                    [
                        1.469543695449829
                    ]
                ]
            ],
            [
                [
                    [
                        -0.36351922154426575
                    ],
                    [
                        -0.5452637672424316
                    ],
                    [
                        -0.78955078125
                    ]
                ]
            ]
        ];

        double[][][][] dy = [
            [
                [
                    [
                        0.8594946265220642
                    ],
                    [
                        -0.368821382522583
                    ],
                    [
                        -1.2551774978637695
                    ]
                ]
            ],
            [
                [
                    [
                        -0.6843253374099731
                    ],
                    [
                        -0.3569713830947876
                    ],
                    [
                        0.1738794893026352
                    ]
                ]
            ],
            [
                [
                    [
                        -1.0914876461029053
                    ],
                    [
                        -1.4308050870895386
                    ],
                    [
                        0.9048557281494141
                    ]
                ]
            ]
        ];

        double[][][][] dx = [
            [
                [
                    [
                        0.32957834005355835
                    ],
                    [
                        0.4049263000488281
                    ],
                    [
                        0.5905399322509766
                    ],
                    [
                        -0.44912227988243103
                    ],
                    [
                        0.03475723788142204
                    ]
                ]
            ],
            [
                [
                    [
                        -0.03147139772772789
                    ],
                    [
                        -0.1618189811706543
                    ],
                    [
                        -0.1659265160560608
                    ],
                    [
                        0.19226835668087006
                    ],
                    [
                        -0.10319620370864868
                    ]
                ]
            ],
            [
                [
                    [
                        -0.1192651093006134
                    ],
                    [
                        -0.799324095249176
                    ],
                    [
                        -0.26175758242607117
                    ],
                    [
                        0.4121021330356598
                    ],
                    [
                        -0.02743864245712757
                    ]
                ]
            ]
        ];

        double[][] dw = [
            [
                -3.378697633743286,
                3.398071765899658,
                -0.2388225644826889,
                -2.162553310394287,
                -0.3848420977592468
            ],
            [
                -3.1748883724212646,
                -0.2556416988372803,
                -0.0072898222133517265,
                -1.1093761920928955,
                2.4745864868164062
            ],
            [
                3.887911319732666,
                -3.293951988220215,
                0.7497283220291138,
                1.8908281326293945,
                1.32449471950531
            ]
        ];

        double[] db = [
            -0.9163183569908142,
            -2.156597852706909,
            -0.17644226551055908
        ];
        
        Test(
            layer: new DenseLinear(5, 3),

            w: Tensor<double>.FromJaggedArray(w).ToFloat(),
            dw: Tensor<double>.FromJaggedArray(dw).ToFloat(),
            b: Tensor<double>.FromJaggedArray(b).ReshapeShared(new TensorShape(b.Length, 1)).ToFloat(),
            db: Tensor<double>.FromJaggedArray(db).ReshapeShared(new TensorShape(db.Length, 1)).ToFloat(),

            x: Tensor<double>.FromJaggedArray(x).ToFloat(),
            dx: Tensor<double>.FromJaggedArray(dx).ToFloat(),

            y: Tensor<double>.FromJaggedArray(y).ReshapeShared(new TensorShape(3, 3)).ToFloat(),
            dy: Tensor<double>.FromJaggedArray(dy).ReshapeShared(new TensorShape(3, 3)).ToFloat()
        );
    }
}