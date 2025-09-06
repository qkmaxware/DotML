using System.Text.RegularExpressions;
using DotML.Network.Initialization;

namespace DotML.Network;

[WorkInProgress]
public abstract class SkipConnection : FeedforwardNetworkLayer {

    /// <summary>
    /// The capture source where the skip connection draws it's features from
    /// </summary>
    public InputCapture CaptureSource {get; private set;}

    /// <summary>
    /// Automatically crop or pad across rows&columns for captured inputs to match input shape
    /// </summary>
    public bool AutoCropOrPad {get; set;} = true;

    public SkipConnection(Shape3D input_shape, Shape3D output_shape, InputCapture captureSource) {
        this.CaptureSource = captureSource;
        this.InputShape = input_shape;
        this.OutputShape = output_shape;
    }

    public override FeatureSet<float> EvaluateSync(FeatureSet<float> channels) {
        return Combine(CaptureSource.CapturedInput, new BatchedFeatureSet<float>(channels))[0];
    }

    public override BatchedFeatureSet<float> EvaluateSync(BatchedFeatureSet<float> features) {
        // TODO AUTO padding / cropping 
        var captured = CaptureSource.CapturedInput;
        if (captured is not null && AutoCropOrPad) {
            var hdiff = features.Columns - captured.Columns;    // + if padding, - if cropping
            var vdiff = features.Rows - captured.Rows;          // + if padding, - if cropping

            if (hdiff != 0 || vdiff != 0) {
                var lpad = hdiff / 2;                           // Attempt to centre the padding/cropping
                var rpad = hdiff - lpad;
                var tpad = vdiff / 2;                           // Attempt to centre the padding/cropping
                var bpad = vdiff - tpad;

                var bs = new FeatureSet<float>[captured.Batches];
                for (var b = 0; b < bs.Length; b++) {
                    var fs = new Matrix<float>[captured.Channels];
                    for (var f = 0; f < fs.Length; f++) {
                        fs[f] = captured[b, f].Pad(
                                  top: tpad, 
                            left: lpad, right: rpad, 
                                bottom: bpad
                        );
                    }
                    bs[b] = new FeatureSet<float>(fs);
                }
                captured = new BatchedFeatureSet<float>(bs);
            }
        }
        return Combine(captured, features);
    }

    /// <summary>
    /// Combine the features from the capture source with the features passed into this connection as input
    /// </summary>
    /// <param name="skipConnection">capture source features</param>
    /// <param name="input">input features</param>
    /// <returns>combined features</returns>
    public abstract BatchedFeatureSet<float> Combine(BatchedFeatureSet<float>? skipConnection, BatchedFeatureSet<float> input);
}