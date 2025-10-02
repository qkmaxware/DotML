using DotML.Network;
using DotML.Network.IO;
using DotML.Network.Training;

namespace DotML.Test.Layers.Convolution;

[TestClass]
public class Conv2DTest
{

    private const float epsilon = 0.001f;

    [TestMethod]
    public void TestSafetensors()
    {
        var saver = new SafetensorSerializer();
        var loader = new SafetensorDeserializer();

        var layer = new Conv2D(
            outChannels: 1,
            inChannelsPerGroup: 1,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (0, 0, 0, 0)
        );

        var tensors = saver.Serialize(layer);

        // Assert that the correct tensors are stored
        Assert.AreEqual(2, tensors.Count);
        Assert.AreEqual(true, tensors.ContainsKey(nameof(Conv2D.Weights)));
        Assert.AreEqual(true, tensors.ContainsKey(nameof(Conv2D.Biases)));

        // Assert that the stored values in the safetensor set are in fact the same as configured in the layer
        Assert.AreEqual(true, layer.Weights.Equals(tensors.GetTensor<float>(nameof(Conv2D.Weights)), 0.0001f));
        Assert.AreEqual(true, layer.Biases.Equals(tensors.GetTensor<float>(nameof(Conv2D.Biases)), 0.0001f));

        // Store and load from file
        var path = "conv2d_test.safetensors";
        tensors.WriteToFile(path);
        var reloadedTensors = Safetensors.ReadFromFile(path);

        // Update the layer weights from the reloaded ones
        loader.Deserialize(layer, reloadedTensors);

        // See that the newly set tensors matched the original ones exactly
        Assert.AreEqual(true, layer.Weights.Equals(tensors.GetTensor<float>(nameof(Conv2D.Weights)), 0.0001f));
        Assert.AreEqual(true, layer.Biases.Equals(tensors.GetTensor<float>(nameof(Conv2D.Biases)), 0.0001f));
    }

    [TestMethod]
    public void TestChannels1Stride1Padding0Kernel3_FFOnly()
    {
        var kernels = new Matrix<float>[]{
            new Matrix<float>(new float[,] {
                {1, 0, 1},
                {0, 1, 0},
                {1, 0, 1}
            })
        };
        var weights = Tensor<float>.FromRectangularArray(new float[,,,] {
            {
                {
                    {1, 0, 1},
                    {0, 1, 0},
                    {1, 0, 1}
                }
            }
        });
        var layer = new Conv2D(
            outChannels: 1,
            inChannelsPerGroup: 1,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (0, 0, 0, 0)
        );
        Assert.AreEqual(layer.Weights.Shape, weights.Shape);
        layer.Weights = weights;

        var inputs = Tensor<float>.FromRectangularArray(new float[,,,] {
            {
                {
                    {1, 1, 1, 0, 0},
                    {0, 1, 1, 1, 0},
                    {0, 0, 1, 1, 1},
                    {0, 0, 1, 1, 0},
                    {0, 1, 1, 0, 0},
                }
            }
        });
        var outputs = layer.Forward(inputs);
        Assert.AreEqual(1, outputs.Shape.Length(^3)); // channels
        var results = Tensor<float>.FromRectangularArray(new float[,,,] {
            {
                {
                    {4, 3, 4},
                    {2, 4, 3},
                    {2, 3, 4}
                }
            }
        });
        Assert.AreEqual(true, outputs.Equals(results, epsilon));
    }

    [TestMethod]
    public void TestChannels1Stride1Padding1Kernel3_FFOnly()
    {
        var kernels = Tensor<float>.FromRectangularArray(new float[,,,] {
            {
                {
                   {1, 0, 1},
                    {0, 1, 0},
                    {1, 0, 1}
                }
            }
        });
        var layer = new Conv2D(
            outChannels: 1,
            inChannelsPerGroup: 1,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (1, 1, 1, 1) // Padding.Same
        );
        Assert.AreEqual(layer.Weights.Shape, kernels.Shape);
        layer.Weights = kernels;

        var inputs = Tensor<float>.FromRectangularArray(new float[,,,] {
            {
                {
                    {1, 1, 1, 0, 0},
                    {0, 1, 1, 1, 0},
                    {0, 0, 1, 1, 1},
                    {0, 0, 1, 1, 0},
                    {0, 1, 1, 0, 0},
                }
            }
        });
        var outputs = layer.Forward(inputs);
        var results = Tensor<float>.FromRectangularArray(new float[,,,] {
            {
                {
                    {2, 2, 3, 1, 1},
                    {1, 4, 3, 4, 1},
                    {1, 2, 4, 3, 3},
                    {1, 2, 3, 4, 1},
                    {0, 2, 2, 1, 1},
                }
            }
        });
        Assert.AreEqual(true, outputs.Equals(results, epsilon));
    }

    [TestMethod]
    public void TestChannels1Stride1Padding1Kernel3()
    {
        var layer = new Conv2D(
            outChannels: 1,
            inChannelsPerGroup: 1,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (1, 1, 1, 1) // same
        );
        var kernel = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            -0.33186790347099304,
            -0.11537425220012665,
            -0.20978042483329773,
            0.08112597465515137,
            -0.16276657581329346,
            0.051479071378707886,
            -0.3236038386821747,
            -0.1491357535123825,
            0.3179304301738739
        ]).ToFloat();
        Assert.AreEqual(layer.Weights.Shape, kernel.Shape);
        layer.Weights = kernel;
        layer.Biases = Tensor<float>.Vec(new float[1] { 0.25102999806404114f });

        var x = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
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
        ]).ToFloat();
        var context = new EvaluationContext();
        var y_projected = layer.Forward(x, context);
        var y_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
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
        ]).ToFloat();
        Assert.AreEqual(true, y_projected.Equals(y_truth, 0.0001f));

        var dy = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
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
        ]).ToFloat();

        var back = layer.Backward(dy, context);
        Assert.IsInstanceOfType<WeightAndBiasGradients>(back);
        var gradients = (WeightAndBiasGradients)back;
        var db_truth = Tensor<double>.Vec([-3.547349214553833]).ToFloat();
        Assert.AreEqual(true, db_truth.Equals(gradients.dB, 0.001f));
        var dw_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            -10.99089527130127,
            -14.076019287109375,
            -7.06515645980835,
            -3.2001407146453857,
            -4.924467086791992,
            8.84919548034668,
            3.4817280769348145,
            -0.5803017616271973,
            -3.469475746154785
        ]).ToFloat();
        Assert.AreEqual(true, dw_truth.Equals(gradients.dW, 0.001f));
        var dx_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
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
        ]).ToFloat();
        Assert.AreEqual(true, gradients.dX.Equals(dx_truth, 0.001f));
    }

    [TestMethod]
    public void TestChannels1Stride2Padding1Kernel3()
    {
        var layer = new Conv2D(
            outChannels: 1,
            inChannelsPerGroup: 1,
            groups: 1,
            kernel: (3, 3),
            stride: (2, 2),
            dilation: (1, 1),
            padding: (1, 1, 1, 1)
        );
        var kernel = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            0.16329312324523926,
            0.09592697024345398,
            0.3095523416996002,
            0.244705468416214,
            -0.004788994789123535,
            0.1564023196697235,
            -0.33284634351730347,
            -0.2813574969768524,
            0.1707456409931183
        ]).ToFloat();
        Assert.AreEqual(layer.Weights.Shape, kernel.Shape);
        layer.Weights = kernel;
        layer.Biases = Tensor<float>.Vec(new float[1] { -0.27856478095054626f });

        var x = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
            0.648898184299469,
            -0.3165184557437897,
            0.5921686291694641,
            0.669906735420227,
            1.8797013759613037,
            0.8009260892868042,
            -0.2469014674425125,
            1.3706653118133545,
            0.026790814474225044,
            0.8981137871742249,
            -1.9287036657333374,
            -0.5320494771003723,
            0.7677139043807983,
            -0.09868572652339935,
            0.22990605235099792,
            -0.48545515537261963,
            1.3519724607467651,
            -0.0021796601358801126,
            -1.567776083946228,
            -0.8130477666854858,
            -0.10570134222507477,
            -0.01946316659450531,
            0.2615739703178406,
            -0.40154287219047546,
            0.1635170876979828
        ]).ToFloat();
        var context = new EvaluationContext();
        var y_projected = layer.Forward(x, context);
        var y_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            -0.5986804962158203,
            -0.5529718399047852,
            -0.38524511456489563,
            0.015289336442947388,
            -1.0454885959625244,
            0.5372989177703857,
            0.0908353254199028,
            -0.6121324896812439,
            -0.7116078734397888
        ]).ToFloat();
        Assert.AreEqual(true, y_projected.Equals(y_truth, 0.0001f));

        var dy = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            1.2390981912612915,
            -0.27579402923583984,
            -1.463151216506958,
            0.8923978209495544,
            -1.5148130655288696,
            0.35458904504776,
            0.7004590630531311,
            1.1337686777114868,
            -0.1760110855102539
        ]).ToFloat();
        var back = layer.Backward(dy, context);
        Assert.IsInstanceOfType<WeightAndBiasGradients>(back);
        var gradients = (WeightAndBiasGradients)back;
        var db_truth = Tensor<double>.Vec([0.8905434608459473]).ToFloat();
        Assert.AreEqual(true, db_truth.Equals(gradients.dB, 0.001f));
        var dw_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            2.19227933883667,
            -1.2425031661987305,
            -1.0914113521575928,
            -0.0733090490102768,
            -4.718402862548828,
            -1.3711528778076172,
            -2.5750069618225098,
            -1.417886734008789,
            3.2680611610412598
        ]).ToFloat();
        Assert.AreEqual(true, dw_truth.Equals(gradients.dW, 0.001f));
        var dx_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
            -0.005934034939855337,
            0.12630951404571533,
            0.0013207761803641915,
            -0.40117594599723816,
            0.007007023319602013,
            -0.26302453875541687,
            0.3322529196739197,
            -0.06771471351385117,
            0.028901897370815277,
            0.44568321108818054,
            -0.004273688420653343,
            -0.23110996186733246,
            0.007254431955516338,
            -0.15015040338039398,
            -0.0016981250373646617,
            -0.18388989567756653,
            1.0585384368896484,
            0.5349630117416382,
            -0.05445203185081482,
            -0.11665049195289612,
            -0.0033544946927577257,
            0.38699281215667725,
            -0.005429612472653389,
            0.13425317406654358,
            0.0008429161971434951
        ]).ToFloat();
        Assert.AreEqual(true, gradients.dX.Equals(dx_truth, 0.001f));
    }

    [TestMethod]
    public void TestChannels1Stride1Padding0Kernel3()
    {
        var layer = new Conv2D(
            outChannels: 1,
            inChannelsPerGroup: 1,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (0, 0, 0, 0) // "Valid" padding
        );

        var kernel = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            0.1519714593887329,
            -0.017021745443344116,
            -0.17113947868347168,
            0.3106711208820343,
            -0.24575552344322205,
            0.30953022837638855,
            -0.23382768034934998,
            -0.22341418266296387,
            -0.255176305770874
        ]).ToFloat();
        Assert.AreEqual(layer.Weights.Shape, kernel.Shape);
        layer.Weights = kernel;
        layer.Biases = Tensor<float>.Vec([-0.058894991874694824f]);

        var x = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
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
        ]).ToFloat();

        var context = new EvaluationContext();
        var y_projected = layer.Forward(x, context);
        var y_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            -0.4346175491809845,
            0.3202958106994629,
            -1.0561137199401855,
            0.11832758784294128,
            0.2308509200811386,
            -0.4351370334625244,
            0.729254424571991,
            -1.4892170429229736,
            0.03990913927555084
        ]).ToFloat();
        Assert.AreEqual(true, y_projected.Equals(y_truth, 0.0001f));

        var dy = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            0.8627731204032898,
            -1.1305592060089111,
            -0.4596337676048279,
            -0.4029831886291504,
            -1.2035309076309204,
            -1.0401866436004639,
            -0.03843408823013306,
            1.1006325483322144,
            1.9555470943450928
        ]).ToFloat();

        var back = layer.Backward(dy, context);
        Assert.IsInstanceOfType<WeightAndBiasGradients>(back);
        var gradients = (WeightAndBiasGradients)back;

        var db_truth = Tensor<double>.Vec([-0.3563750982284546]).ToFloat();
        Assert.AreEqual(true, db_truth.Equals(gradients.dB, 0.001f));

        var dw_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            0.577020525932312,
            6.409263610839844,
            -0.10182112455368042,
            -1.225755214691162,
            -0.3377056121826172,
            -6.066802024841309,
            1.615986704826355,
            -1.4229464530944824,
            -1.9790031909942627
        ]).ToFloat();
        Assert.AreEqual(true, dw_truth.Equals(gradients.dW, 0.001f));

        var dx_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
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
        ]).ToFloat();
        Assert.AreEqual(true, gradients.dX.Equals(dx_truth, 0.001f));
    }

    [TestMethod]
    public void TestChannels1Stride2Padding0Kernel3()
    {
        var layer = new Conv2D(
            outChannels: 1,
            inChannelsPerGroup: 1,
            groups: 1,
            kernel: (3, 3),
            stride: (2, 2),
            dilation: (1, 1),
            padding: (0, 0, 0, 0) // Valid padding
        );

        var kernel = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            -0.22670862078666687,
            0.29332056641578674,
            -0.19603630900382996,
            -0.09961383044719696,
            0.2943679988384247,
            -0.22048604488372803,
            0.3319796621799469,
            0.1491285264492035,
            -0.25655075907707214
        ]).ToFloat();
        Assert.AreEqual(layer.Weights.Shape, kernel.Shape);
        layer.Weights = kernel;
        layer.Biases = Tensor<float>.Vec([-0.21968214213848114f]);

        var x = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
            -1.187608242034912,
            -0.013705188408493996,
            0.773604691028595,
            -0.019701238721609116,
            1.003239631652832,
            -0.453906774520874,
            0.446717768907547,
            0.8841092586517334,
            -0.908335268497467,
            -0.4518772065639496,
            0.9588450789451599,
            1.2185893058776855,
            -0.35732653737068176,
            0.8467770218849182,
            -1.6297091245651245,
            0.9893779158592224,
            0.15075571835041046,
            -0.7110050916671753,
            0.3123204708099365,
            1.2634272575378418,
            -0.8262707591056824,
            -1.0932413339614868,
            -0.26137712597846985,
            1.3187628984451294,
            0.039951782673597336
        ]).ToFloat();

        var context = new EvaluationContext();
        var y_projected = layer.Forward(x, context);
        var y_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 2, 2), [
            0.46738120913505554,
            -0.4275803565979004,
            -0.277267724275589,
            0.4130247235298157
        ]).ToFloat();
        Assert.AreEqual(true, y_projected.Equals(y_truth, 0.0001f));

        var dy = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 2, 2), [
            0.13445043563842773,
            1.81400728225708,
            -1.0690683126449585,
            -0.22659282386302948
        ]).ToFloat();

        var back = layer.Backward(dy, context);
        Assert.IsInstanceOfType<WeightAndBiasGradients>(back);
        var gradients = (WeightAndBiasGradients)back;

        var db_truth = Tensor<double>.Vec([0.6527965664863586]).ToFloat();
        Assert.AreEqual(true, db_truth.Equals(gradients.dB, 0.001f));

        var dw_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 3, 3), [
            0.2995468080043793,
            -1.5322096347808838,
            2.675182342529297,
            0.6461487412452698,
            -1.8196029663085938,
            -0.22701019048690796,
            0.42329028248786926,
            2.569827079772949,
            -2.7339699268341064
        ]).ToFloat();
        Assert.AreEqual(true, dw_truth.Equals(gradients.dW, 0.001f));

        var dx_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
            -0.030481072142720222,
            0.039437077939510345,
            -0.4376082718372345,
            0.532085657119751,
            -0.35561129450798035,
            -0.01339312270283699,
            0.0395779050886631,
            -0.21034465730190277,
            0.5339856743812561,
            -0.39996328949928284,
            0.2870018184185028,
            -0.29352930188179016,
            0.8286668658256531,
            0.20405590534210205,
            -0.4209645092487335,
            0.1064939871430397,
            -0.3146995007991791,
            0.25828641653060913,
            -0.06670167297124863,
            0.049960557371377945,
            -0.35490894317626953,
            -0.15942858159542084,
            0.1990460902452469,
            -0.03379145264625549,
            0.058132562786340714
        ]).ToFloat();
        Assert.AreEqual(true, gradients.dX.Equals(dx_truth, 0.001f));
    }

    [TestMethod]
    public void TestChannels3Stride1PaddingSameKernel3()
    {
        var layer = new Conv2D(
            outChannels: 1,
            inChannelsPerGroup: 3,
            groups: 1,
            kernel: (3, 3),
            stride: (1, 1),
            dilation: (1, 1),
            padding: (1, 1, 1, 1) // SAME padding
        );

        var weights = Tensor<double>.FromFlattenedArray(new TensorShape(1, 3, 3, 3), [
            // Filter for channel 0
            -0.03724499046802521,
            -0.12150624394416809,
            -0.1883133053779602,
            -0.15678884088993073,
            -0.14671427011489868,
            -0.09504218399524689,
            0.16976715624332428,
            -0.05292584002017975,
            0.06834743916988373,

            // Filter for channel 1
            0.1203652173280716,
            -0.05575203895568848,
            -0.11788377165794373,
            0.040189072489738464,
            0.025411456823349,
            0.027559593319892883,
            -0.12857350707054138,
            -0.1310662031173706,
            -0.13793152570724487,

            // Filter for channel 2
            0.02665373682975769,
            -0.0669141411781311,
            -0.048021718859672546,
            -0.043146297335624695,
            0.17804841697216034,
            0.1263914555311203,
            -0.003790155053138733,
            0.024605125188827515,
            0.08892489969730377
        ]).ToFloat();

        Assert.AreEqual(layer.Weights.Shape, weights.Shape);
        layer.Weights = weights;
        layer.Biases = Tensor<float>.Vec([0.14321748912334442f]);

        var x = Tensor<double>.FromFlattenedArray(new TensorShape(1, 3, 5, 5), [
            // Channel 0
            0.04129666090011597, -1.3909366130828857, 0.21071363985538483, -0.07865647226572037, 1.3091665506362915,
            -0.8029050827026367, -1.1879446506500244, 1.5709104537963867, 2.150278091430664, 0.2091362476348877,
            0.2787802815437317, 1.5367063283920288, 0.2909795343875885, -0.5324845314025879, -0.32815849781036377,
            0.6322647929191589, 1.2863354682922363, -0.9702771306037903, 0.39146965742111206, -1.1259630918502808,
            -0.5308622121810913, 0.48084360361099243, -0.21467261016368866, 0.3157433271408081, -0.3441010117530823,

            // Channel 1
            2.062704086303711, 0.8704235553741455, 0.031899549067020416, -0.31951576471328735, 0.8669164776802063,
            -0.01013757660984993, -0.04607278108596802, 1.1386950016021729, 1.0154192447662354, -0.5821844339370728,
            1.2897073030471802, -0.2599606513977051, 1.860145092010498, 1.2377678155899048, 0.8810083866119385,
            0.01574167236685753, 2.04868483543396, -0.9194207191467285, 2.2229480743408203, 1.3054641485214233,
            0.7976852655410767, -2.060070276260376, 0.22221124172210693, 0.012128844857215881, 0.7650163173675537,

            // Channel 2
            -0.6780450940132141, 1.0030226707458496, 1.5268474817276, -0.2175118327140808, 2.3705554008483887,
            1.9843108654022217, 0.3084131181240082, 0.379267156124115, -0.1688816249370575, -0.28296080231666565,
            -0.0002048702008323744, -0.5540425181388855, -0.18671707808971405, 0.21668469905853271, -0.4144744575023651,
            0.07831291854381561, -0.9285159707069397, 0.5196751356124878, 0.5575506687164307, -0.4535541534423828,
            -0.7316989302635193, -0.6764885783195496, -0.7969027757644653, 0.019437022507190704, -0.38583800196647644
        ]).ToFloat();

        var context = new EvaluationContext();
        var y_projected = layer.Forward(x, context);

        var y_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
            //[
                0.39704465866088867, 0.7455122470855713, 0.1378490924835205, 0.146591916680336, 0.6975279450416565,
                0.7090077996253967, 0.0737203061580658, -0.055984120815992355, -1.3476455211639404, -1.0077036619186401,
                -0.21934619545936584, -0.5674827098846436, -0.8373020887374878, -0.4876246452331543, -0.01668531820178032,
                -0.29384809732437134, -0.46168583631515503, 0.41740116477012634, 0.3439185917377472, 0.4182371199131012,
                -0.5989762544631958, 0.0107339546084404, -0.1719706505537033, -0.0069081708788871765, 0.4568127691745758
            //]
        ]).ToFloat();
        Assert.AreEqual(true, y_projected.Equals(y_truth, 0.0001f));

        var dy = Tensor<double>.FromFlattenedArray(new TensorShape(1, 1, 5, 5), [
            //[
                0.29369616508483887, -0.7291655540466309, -1.1719928979873657, 0.2644590437412262, 0.02122787944972515,
                -1.6344319581985474, -0.7068682312965393, 0.13566316664218903, 1.7301790714263916, 0.5853704214096069,
                0.864045262336731, -0.32174479961395264, -1.3167892694473267, 2.5236117839813232, -0.4630984365940094,
                -1.4411110877990723, 0.9561700820922852, 0.8640719652175903, -1.2451316118240356, 1.6089839935302734,
                -0.8261396288871765, 1.1503515243530273, 0.8561965823173523, -2.084163188934326, -0.7908592224121094
            //]
        ]).ToFloat();

        var back = layer.Backward(dy, context);
        var gradients = (WeightAndBiasGradients)back;

        var db_truth = Tensor<float>.Vec([-0.8774690628051758f]);
        Assert.AreEqual(true, gradients.dB.Equals(db_truth, 0.001f));

        var dw_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 3, 3, 3), [
            // Channel 0
            8.808441162109375, 6.179363250732422, -0.9415181875228882,
            8.45761489868164, 2.7690742015838623, 1.7702165842056274,
            -1.201981782913208, -0.2414613962173462, -9.446768760681152,

            // Channel 1
            3.6098437309265137, -7.421334266662598, -4.642982006072998,
            9.279187202453613, -0.04291725158691406, -4.1553850173950195,
            -3.946850299835205, 3.7966060638427734, 3.2868776321411133,

            // Channel 2
            0.9937957525253296, -0.6802810430526733, 5.116835594177246,
            -1.1801586151123047, -7.999062538146973, 0.3729352355003357,
            0.007214784622192383, 0.7348756790161133, -1.8511812686920166
        ]).ToFloat();
        Assert.AreEqual(true, gradients.dW.Equals(dw_truth, 0.001f));

        var dx_truth = Tensor<double>.FromFlattenedArray(new TensorShape(1, 3, 5, 5), [
            // Channel 0
            //[
                0.2961565852165222, 0.651442289352417, 0.25197362899780273, -0.18831628561019897, -0.42519110441207886,
                0.11828756332397461, 0.02290293201804161, -0.0403105728328228, -0.49042844772338867, -0.6523308753967285,
                0.029670335352420807, 0.24329259991645813, -0.17234298586845398, -0.226764515042305, -0.045661091804504395,
                0.018699035048484802, -0.3023488223552704, 0.21066761016845703, -0.33243662118911743, 0.5678414106369019,
                0.17944248020648956, -0.22690823674201965, -0.0999370589852333, 0.7465089559555054, 0.14385534822940826,
            //],
            // Channel 1
            //[
                -0.01580066606402397, 0.19087515771389008, 0.24476896226406097, -0.06672148406505585, -0.22876781225204468,
                -0.1015831008553505, -0.0942230224609375, 0.6887791156768799, 0.15428510308265686, -0.2483755350112915,
                0.5095638036727905, 0.48393914103507996, -0.39440488815307617, -0.1502920687198639, -0.2005101442337036,
                0.11444885283708572, 0.24791280925273895, -0.5434392690658569, -0.11268208920955658, 0.008963286876678467,
                0.09118097275495529, 0.003230080008506775, -0.11534585058689117, -0.2240099161863327, -0.11667610704898834,
            //],
            // Channel 2
            //[
                0.1742788702249527, 0.08726471662521362, -0.24125923216342926, -0.20864519476890564, -0.08505077660083771,
                -0.3169117271900177, -0.3807316720485687, -0.06369240581989288, 0.0841788649559021, 0.25674307346343994,
                0.2521039545536041, -0.026258014142513275, -0.5870032906532288, 0.4399997889995575, 0.3568977117538452,
                -0.18942174315452576, 0.010247036814689636, 0.0897611752152443, -0.1578858643770218, 0.4951251745223999,
                -0.23580901324748993, -0.044439367949962616, 0.4987701177597046, -0.1886407732963562, -0.47536560893058777
            //]
        ]).ToFloat();
        Assert.AreEqual(true, gradients.dX.Equals(dx_truth, 0.001f));
    }


}
