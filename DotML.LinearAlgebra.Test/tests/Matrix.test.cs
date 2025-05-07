namespace DotML.Test;

[TestClass]
public class TestMatrix {
    [TestMethod]
    public void TestCreation() {
        Matrix<double> matrix = new Matrix<double>(5, 2);
        Assert.AreEqual(5, matrix.Rows);
        Assert.AreEqual(2, matrix.Columns);

        Matrix<double> m2 = new Matrix<double>(new double[,]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Assert.AreEqual(2, m2.Rows);
        Assert.AreEqual(3, m2.Columns);
        Assert.AreEqual(1, m2[0, 0]);
        Assert.AreEqual(3, m2[0, 2]);
        Assert.AreEqual(4, m2[1, 0]);
        Assert.AreEqual(6, m2[1, 2]);

        var zero = Matrix<double>.Zeros(3);
        Assert.AreEqual(3, zero.Rows);
        Assert.AreEqual(3, zero.Columns);
        foreach (var element in zero)
            Assert.AreEqual(0, element);

        var ones = Matrix<double>.Ones(4);
        Assert.AreEqual(4, ones.Rows);
        Assert.AreEqual(4, ones.Columns);
        foreach (var element in ones)
            Assert.AreEqual(1, element);
    }

    [TestMethod]
    public void TestTranspose() {
        Matrix<double> m2 = new Matrix<double>(new double[,]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Assert.AreEqual(2, m2.Rows);
        Assert.AreEqual(3, m2.Columns);
        Assert.AreEqual(1, m2[0, 0]);
        Assert.AreEqual(3, m2[0, 2]);
        Assert.AreEqual(4, m2[1, 0]);
        Assert.AreEqual(6, m2[1, 2]);
        var m2shape = m2.Shape;
        Assert.AreEqual(2, m2shape.Rows);
        Assert.AreEqual(3, m2shape.Columns);

        var transposed = m2.Transpose();
        Assert.AreEqual(3, transposed.Rows);
        Assert.AreEqual(2, transposed.Columns);
        Assert.AreEqual(1, transposed[0, 0]);
        Assert.AreEqual(4, transposed[0, 1]);
        Assert.AreEqual(3, transposed[2, 0]);
        Assert.AreEqual(6, transposed[2, 1]);
        var tshape = transposed.Shape;
        Assert.AreEqual(3, tshape.Rows);
        Assert.AreEqual(2, tshape.Columns);
    }

    [TestMethod]
    public void TestMap() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        var B = A.Transform((x) => x * x);
        Assert.AreEqual(1, B[0,0]);
        Assert.AreEqual(2*2, B[0,1]);
        Assert.AreEqual(3*3, B[1,0]);
        Assert.AreEqual(4*4, B[1,1]);

        var C = A.Transform<float>((x) => (float)(x * x));
    }

    [TestMethod]
    public void TestFlatten() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        var row_order = A.FlattenRows().ToArray();
        Assert.AreEqual(A.Size, row_order.Length);
        Assert.AreEqual(A[0,0], row_order[0]);  // 1
        Assert.AreEqual(A[0,1], row_order[1]);  // 2
        Assert.AreEqual(A[1,0], row_order[2]);  // 3
        Assert.AreEqual(A[1,1], row_order[3]);  // 4

        var col_order = A.FlattenColumns().ToArray();
        Assert.AreEqual(A.Size, col_order.Length);
        Assert.AreEqual(A[0,0], col_order[0]);  // 1
        Assert.AreEqual(A[1,0], col_order[1]);  // 3
        Assert.AreEqual(A[0,1], col_order[2]);  // 2
        Assert.AreEqual(A[1,1], col_order[3]);  // 4
    }

    [TestMethod]
    public void TestExtract() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2, 3},
            { 4, 5, 6}
        });

        var row0 = A.ExtractRowVector(0);
        var row1 = A.ExtractRowVector(1);
        Assert.AreEqual(3, row0.Dimensionality);
        Assert.AreEqual(3, row1.Dimensionality);
        Assert.AreEqual(1, row0[0]); Assert.AreEqual(2, row0[1]); Assert.AreEqual(3, row0[2]);
        Assert.AreEqual(4, row1[0]); Assert.AreEqual(5, row1[1]); Assert.AreEqual(6, row1[2]);

        var col0 = A.ExtractColumnVector(0);
        var col1 = A.ExtractColumnVector(1);
        var col2 = A.ExtractColumnVector(2);
        Assert.AreEqual(2, col0.Dimensionality);
        Assert.AreEqual(2, col1.Dimensionality);
        Assert.AreEqual(2, col2.Dimensionality);
        Assert.AreEqual(1, col0[0]); Assert.AreEqual(4, col0[1]); 
        Assert.AreEqual(2, col1[0]); Assert.AreEqual(5, col1[1]); 
        Assert.AreEqual(3, col2[0]); Assert.AreEqual(6, col2[1]); 
    }

    [TestMethod]
    public void TestElementWise() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });
        Matrix<double> B = new Matrix<double>(new double[,] {
            { 5, 6 },
            { 7, 8 }
        });

        var C = A.ElementWise(B, (a, b) => a + b);
        Assert.AreEqual(1+5, C[0,0]);
        Assert.AreEqual(2+6, C[0,1]);
        Assert.AreEqual(3+7, C[1,0]);
        Assert.AreEqual(4+8, C[1,1]);
    }

    [TestMethod]
    public void TestReshape() {
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });
        var row = A.Reshape(new Shape2D(rows: 1, columns: 4)).ToArray();
        var col = A.Reshape(new Shape2D(rows: 4, columns: 1)).ToArray();

        Assert.AreEqual(1, row.Length);
        Assert.AreEqual(1, row[0].Rows);
        Assert.AreEqual(4, row[0].Columns);
        
        Assert.AreEqual(1, col.Length);
        Assert.AreEqual(4, col[0].Rows);
        Assert.AreEqual(1, col[0].Columns);
    }

    [TestMethod]
    public void TestCompatibleMultiply() {
        // Arrange: Define two matrices to multiply
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Matrix<double> B = new Matrix<double>(new double[,] {
            { 5, 6 },
            { 7, 8 }
        });

        // Expected result of A * B
        Matrix<double> expected = new Matrix<double>(new double[,] {
            { 19, 22 },
            { 43, 50 }
        });

        // Act: Multiply matrices A and B
        var result = A * B;

        // Assert: Verify the result matches the expected output
        Assert.AreEqual(expected.Rows, result.Rows);
        Assert.AreEqual(expected.Columns, result.Columns);
        foreach (var pair in expected.Zip(result)) {
            Assert.AreEqual(pair.First, pair.Second, 0.01);
        }
    }

    [TestMethod]
    public void TestCompatibleMultiplyVector() {
        // Arrange: Define two matrices to multiply
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Vec<double> B = new double[] {
            5,
            6 
        };

        // Expected result of A * B
        Vec<double> expected = new double[] {
            17,
            39
        };

        // Act: Multiply matrices A and B
        var result = A * B;

        // Assert: Verify the result matches the expected output
        Assert.AreEqual(expected.Dimensionality, result.Dimensionality);
        foreach (var pair in expected.Zip(result)) {
            Assert.AreEqual(pair.First, pair.Second, 0.01);
        }
    }

    [TestMethod]
    public void TestCompatibleMultiplyTransposed() {
        // Arrange: Define two matrices to multiply
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Matrix<double> B = new Matrix<double>(new double[,] {
            { 5, 6 },
            { 7, 8 }
        });

        // Expected result of A * B
        Matrix<double> expected = A.Transpose() * B;

        // Act: Multiply matrices A and B
        var result = A.MultiplyTransposedWith(B);

        // Assert: Verify the result matches the expected output
        Assert.AreEqual(expected.Rows, result.Rows);
        Assert.AreEqual(expected.Columns, result.Columns);
        foreach (var pair in expected.Zip(result)) {
            Assert.AreEqual(pair.First, pair.Second, 0.01);
        }
    }

    [TestMethod]
    public void TestIncompatibleMultiply() {
        // Arrange: Define two incompatible matrices
        Matrix<double> A = new Matrix<double>(new double[,] {
            { 1, 2, 3 }
        }); // 1x3 matrix

        Matrix<double> B = new Matrix<double>(new double[,] {
            { 4, 5 },
            { 6, 7 },
            { 8, 9 }
        }); // 3x2 matrix

        // Act & Assert: Try to multiply the matrices (this should throw an exception)
        Assert.ThrowsException<ArithmeticException>(() => B*A);
    }

    [TestMethod]
    public void TestConvolve() {
        var kernels = new Matrix<double>(new double[,] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            });
        var input = new Matrix<double>(new double[,]{
            {1, 1, 1, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 1, 1, 1},
            {0, 0, 1, 1, 0},
            {0, 1, 1, 0, 0},
        });
        var output = input.Convolve(kernels);

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
    public void TestTransposeConvolveInPadding0OutPadding0Stride1() {
        Matrix<double> input = new Matrix<double>(new double[,]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        Matrix<double> kernel = new Matrix<double>(new double[,]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        Matrix<double> result_truth = new Matrix<double>(new double[,]{
            {1, 4, 10, 12, 9},
            {8, 26, 56, 54, 36},
            {30, 84, 165, 144, 90},
            {56, 134, 236, 186, 108},
            {49, 112, 190, 144, 81}
        });
        var result_predicted = input.TransposeConvolve(kernel);
        Assert.AreEqual(result_truth.Rows, result_predicted.Rows);
        Assert.AreEqual(result_truth.Columns, result_predicted.Columns);

        foreach (var (predicted, truth) in result_predicted.Zip(result_truth)) {
            Assert.AreEqual(truth, predicted, 0.0001);
        }
    }

    [TestMethod]
    public void TestTransposeConvolveInPadding1OutPadding0Stride2() {
        // nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False)
        Matrix<double> input = Matrix<double>.FromJagged(
            [
                [
                    0.09351952373981476,
                    -1.1028000116348267,
                    0.7385574579238892
                ],
                [
                    1.25326406955719,
                    -0.4579811096191406,
                    0.2503132224082947
                ],
                [
                    1.4259989261627197,
                    0.4574931561946869,
                    2.518425941467285
                ]
            ]
        );

        Matrix<double> kernel = Matrix<double>.FromJagged(
            [
                [
                    0.22182723879814148,
                    0.25718554854393005,
                    -0.13989269733428955
                ],
                [
                    -0.2033747434616089,
                    -0.18745946884155273,
                    0.27698227763175964
                ],
                [
                    0.19349196553230286,
                    0.08252885937690735,
                    -0.23545603454113007
                ]
            ]
        );

        Matrix<double> result_truth = Matrix<double>.FromJagged(
            [
                [
                    -0.017531121149659157,
                    0.25018492341041565,
                    0.20673030614852905,
                    -0.45565998554229736,
                    -0.13844959437847137
                ],
                [
                    0.33003947138786316,
                    -0.5123178362846375,
                    -0.20879894495010376,
                    0.5221604108810425,
                    0.12532925605773926
                ],
                [
                    -0.23493622243404388,
                    0.4402737319469452,
                    0.08585289865732193,
                    -0.1777600347995758,
                    -0.04692358523607254
                ],
                [
                    0.4701767563819885,
                    -0.4817066490650177,
                    0.07986396551132202,
                    0.6509235501289368,
                    0.6683608293533325
                ],
                [
                    -0.267316997051239,
                    0.3019338846206665,
                    -0.0857614204287529,
                    -0.38546669483184814,
                    -0.4721027910709381
                ]
            ]
        );
        var result_predicted = input.TransposeConvolve(kernel, 
            flip_kernel: false, 
            outputStrideX: 2, outputStrideY: 2, 
            inputPaddingX: 1,
            inputPaddingY: 1,
            outputPaddingX: 0,
            outputPaddingY: 0,
            bias: 0
        );
        Assert.AreEqual(result_truth.Rows, result_predicted.Rows);
        Assert.AreEqual(result_truth.Columns, result_predicted.Columns);

        foreach (var (predicted, truth) in result_predicted.Zip(result_truth)) {
            Assert.AreEqual(truth, predicted, 0.0001);
        }
    }

    [TestMethod]
    public void TestTransposeConvolveInPadding0OutPadding1Stride2() {
        // nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0, output_padding=1, bias=False)
        Matrix<double> input = Matrix<double>.FromJagged(
            [
                [
                    0.43177855014801025,
                    -1.1046756505966187,
                    -0.7093443870544434
                ],
                [
                    -0.3726115822792053,
                    0.5411018133163452,
                    0.1476650834083557
                ],
                [
                    -0.5250610709190369,
                    -0.9599573016166687,
                    1.7568827867507935
                ]
            ]
        );

        Matrix<double> kernel = Matrix<double>.FromJagged(
            [
                [
                    0.07606670260429382,
                    0.3209017813205719,
                    0.24665507674217224
                ],
                [
                    -0.20184862613677979,
                    0.20720753073692322,
                    -0.08361458778381348
                ],
                [
                    -0.31324315071105957,
                    -0.08871690928936005,
                    -0.2839755415916443
                ]
            ]
        );

        Matrix<double> result_truth = Matrix<double>.FromJagged(
            [
                [
                    0.03284396976232529,
                    0.1385585069656372,
                    0.022471338510513306,
                    -0.3544923961162567,
                    -0.32643136382102966,
                    -0.2276298701763153,
                    -0.17496339976787567,
                    0.0
                ],
                [
                    -0.08715390413999557,
                    0.08946776390075684,
                    0.18687428534030914,
                    -0.2288971096277237,
                    0.23554718494415283,
                    -0.14698149263858795,
                    0.05931153893470764,
                    0.0
                ],
                [
                    -0.16359500586986542,
                    -0.15787777304649353,
                    0.17267082631587982,
                    0.2716439366340637,
                    0.6805959939956665,
                    0.11031682789325714,
                    0.23785880208015442,
                    0.0
                ],
                [
                    0.07521113753318787,
                    -0.07720792293548584,
                    -0.07806489616632462,
                    0.1121203675866127,
                    -0.0750499963760376,
                    0.030597317963838577,
                    -0.012346955016255379,
                    0.0
                ],
                [
                    0.0767783597111702,
                    -0.13543608784675598,
                    -0.26621362566947937,
                    -0.3560568690299988,
                    -0.3030528426170349,
                    0.5506864190101624,
                    0.39141079783439636,
                    0.0
                ],
                [
                    0.10598285496234894,
                    -0.1087966114282608,
                    0.23766882717609406,
                    -0.19891038537025452,
                    -0.27435797452926636,
                    0.3640393316745758,
                    -0.14690102636814117,
                    0.0
                ],
                [
                    0.1644717901945114,
                    0.04658179357647896,
                    0.44980454444885254,
                    0.08516444265842438,
                    -0.27772706747055054,
                    -0.15586520731449127,
                    -0.4989117383956909,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]
            ]
        );
        var result_predicted = input.TransposeConvolve(kernel, 
            flip_kernel: false, 
            outputStrideX: 2, outputStrideY: 2, 
            inputPaddingX: 0,
            inputPaddingY: 0,
            outputPaddingX: 1,
            outputPaddingY: 1,
            bias: 0
        );
        Assert.AreEqual(result_truth.Rows, result_predicted.Rows);
        Assert.AreEqual(result_truth.Columns, result_predicted.Columns);

        foreach (var (predicted, truth) in result_predicted.Zip(result_truth)) {
            Assert.AreEqual(truth, predicted, 0.0001);
        }
    }

    [TestMethod]
    public void TestTransposeConvolveInPadding1OutPadding1Stride2() {
        // nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0, output_padding=1, bias=False)
        Matrix<double> input = Matrix<double>.FromJagged(
            [
                [
                    -1.992504358291626,
                    0.27014362812042236,
                    0.22134125232696533
                ],
                [
                    0.4154917895793915,
                    1.6616926193237305,
                    -1.271235704421997
                ],
                [
                    0.36543068289756775,
                    0.6304700374603271,
                    0.06103163957595825
                ]
            ]
        );

        Matrix<double> kernel = Matrix<double>.FromJagged(
            [
                [
                    -0.27457696199417114,
                    0.0762561559677124,
                    -0.2322739064693451
                ],
                [
                    -0.21782729029655457,
                    -0.059218764305114746,
                    -0.02875646948814392
                ],
                [
                    0.17116573452949524,
                    0.058541566133499146,
                    0.281290739774704
                ]
            ]
        );

        Matrix<double> result_truth = Matrix<double>.FromJagged(
            [
                [
                    0.11799364537000656,
                    -0.0015472657978534698,
                    -0.015997571870684624,
                    -0.05598254129290581,
                    -0.013107554987072945,
                    -0.006364992819726467
                ],
                [
                    -0.08496052026748657,
                    -1.0670040845870972,
                    0.14252892136573792,
                    0.07695913314819336,
                    -0.08398188650608063,
                    0.3575361371040344
                ],
                [
                    -0.024604910984635353,
                    -0.3739100694656372,
                    -0.09840338677167892,
                    0.22912541031837463,
                    0.07528100907802582,
                    0.03655625134706497
                ],
                [
                    0.05218987911939621,
                    0.14330627024173737,
                    0.14535531401634216,
                    0.08662715554237366,
                    -0.0697660967707634,
                    -0.3717629015445709
                ],
                [
                    -0.0216403529047966,
                    -0.14784207940101624,
                    -0.037335656583309174,
                    -0.031424447894096375,
                    -0.003614218207076192,
                    -0.0017550544580444694
                ],
                [
                    0.021392883732914925,
                    0.21070712804794312,
                    0.03690870478749275,
                    0.18779189884662628,
                    0.0035728877410292625,
                    0.01716763526201248
                ]
            ]
        );
        var result_predicted = input.TransposeConvolve(kernel, 
            flip_kernel: false, 
            outputStrideX: 2, outputStrideY: 2, 
            inputPaddingX: 1,
            inputPaddingY: 1,
            outputPaddingX: 1,
            outputPaddingY: 1,
            bias: 0
        );
        Assert.AreEqual(result_truth.Rows, result_predicted.Rows);
        Assert.AreEqual(result_truth.Columns, result_predicted.Columns);

        foreach (var (predicted, truth) in result_predicted.Zip(result_truth)) {
            Assert.AreEqual(truth, predicted, 0.0001);
        }
    }
    
    [TestMethod]
    public void TestPositivePad() {
        var matrix = Matrix<double>.FromJagged(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        );
        Assert.AreEqual(3, matrix.Rows);
        Assert.AreEqual(3, matrix.Columns);

        var pad = matrix.Pad(top: 4, value: 0);
        Assert.AreEqual(3 + 4, pad.Rows);
        Assert.AreEqual(3, pad.Columns);
        Assert.AreEqual(
            Matrix<double>.FromJagged(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ), 
            pad
        );
        pad = matrix.Pad(bottom: 4, value: 0);
        Assert.AreEqual(3 + 4, pad.Rows);
        Assert.AreEqual(3, pad.Columns);
        Assert.AreEqual(
            Matrix<double>.FromJagged(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ), pad
        );
        pad = matrix.Pad(left: 4, value: 0);
        Assert.AreEqual(3, pad.Rows);
        Assert.AreEqual(3 + 4, pad.Columns);
        Assert.AreEqual(
            Matrix<double>.FromJagged(
                [
                    [0,0,0,0, 1, 2, 3],
                    [0,0,0,0, 4, 5, 6],
                    [0,0,0,0, 7, 8, 9],
                ]
            ),
            pad
        );
        pad = matrix.Pad(right: 4, value: 0);
        Assert.AreEqual(3, pad.Rows);
        Assert.AreEqual(3 + 4, pad.Columns);
        Assert.AreEqual(
            Matrix<double>.FromJagged(
                [
                    [1, 2, 3, 0,0,0,0 ],
                    [4, 5, 6, 0,0,0,0 ],
                    [7, 8, 9, 0,0,0,0 ],
                ]
            ),
            pad
        );
    }

    [TestMethod]
    public void TestNegativePad() {
        var matrix = Matrix<double>.FromJagged(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        );
        Assert.AreEqual(3, matrix.Rows);
        Assert.AreEqual(3, matrix.Columns);

        var crop = matrix.Pad(top: -1);
        Assert.AreEqual(3 - 1, crop.Rows);
        Assert.AreEqual(3, crop.Columns);
        Assert.AreEqual(
            Matrix<double>.FromJagged(
                [
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ),
            crop
        );

        crop = matrix.Pad(bottom: -1);
        Assert.AreEqual(3 - 1, crop.Rows);
        Assert.AreEqual(3, crop.Columns);
        Assert.AreEqual(
            Matrix<double>.FromJagged(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            crop
        );

        crop = matrix.Pad(left: -1);
        Assert.AreEqual(3, crop.Rows);
        Assert.AreEqual(3 - 1, crop.Columns);
        Assert.AreEqual(
            Matrix<double>.FromJagged(
                [
                    [2, 3],
                    [5, 6],
                    [8, 9],
                ]
            ),
            crop
        );

        crop = matrix.Pad(right: -1);
        Assert.AreEqual(3, crop.Rows);
        Assert.AreEqual(3 - 1, crop.Columns);
        Assert.AreEqual(
            Matrix<double>.FromJagged(
                [
                    [1, 2],
                    [4, 5],
                    [7, 8],
                ]
            ),
            crop
        );
    }
}