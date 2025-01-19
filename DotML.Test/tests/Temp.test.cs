using System.Runtime.CompilerServices;
using DotML;
using DotML.Network;
using static DotML.Network.Training.BatchTrainerEnumerator<DotML.Network.FeedforwardNetwork>;

[TestClass]
public class TempTests {
    [TestMethod]
    public void DumpNetwork() {
        var network = VGGNet.Make(VGGNet.Version.VGG16, 3);

        using var writer = new StreamWriter(network.Name + ".md");
        writer.Write(network.ToMarkdown());
    }
    /*
    // dx = dy_0 * w'
        var (batch_count, channel_count, input_height, input_width) = args.InputBatch.Shape;
        var filter_count = layer.FilterCount;
        var (_, _, output_height, output_width) = args.OutputBatch.Shape;
        var stride_x = layer.StrideX;
        var stride_y = layer.StrideY;

        var padding_rows = layer.RowsPadding;
        var padding_cols = layer.ColumnsPadding;

        var result_features = new FeatureSet<double>[batch_count];

        for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) {
            var batch_inputs = args.InputBatch[batchIndex];
            var batch_outputs = args.OutputBatch[batchIndex];
            var batch_errors = args.OutputErrors[batchIndex];

            var batch_features = new Matrix<double>[channel_count];
            for (var channelIndex = 0; channelIndex < channel_count; channelIndex++) {
                var input_error = new double[input_height, input_width];

                for (var filterIndex = 0; filterIndex < filter_count; filterIndex++) {
                    var filter = layer.Filters[filterIndex];
                    var filter_width = filter.Width;
                    var filter_width_m1 = filter_width - 1;
                    var filter_height = filter.Height;
                    var filter_height_m1 = filter_height - 1;

                    var input_padding_rows = (filter_height - 1) / 2;
                    var input_padding_cols = (filter_width - 1) / 2;

                    var padded_output_height = output_height + 2*filter_height_m1;
                    var padded_output_width  = output_width + 2*filter_width_m1;
                    
                    var kernel = filter[channelIndex];
                    var error = batch_errors[filterIndex];
                    
                    // Slide kernel over output
                    for (var outY = -filter_height_m1; outY < padded_output_height; outY++) {
                        var startY = (outY + padding_rows + 1) * stride_y; // TODO verify this computation

                        for (var outX = -filter_width_m1; outX < padded_output_width; outX++) {
                            var startX = (outX + padding_cols + 1) * stride_x; // TODO verify this computation

                            var str = "";
                            var sum = 0.0;
                            for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                                var inv_kernelY = filter_height_m1 - kernelY;
                                var outY_plus_kernel = outY + kernelY;

                                for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                    // Multiply error values by kernel, sum to appropriate position in input
                                    var inv_kernelX = filter_width_m1 - kernelX;
                                    var outX_plus_kernel = outX + kernelX;

                                    var kernel_value = kernel[inv_kernelY, inv_kernelX];
                                    var output_value = error[outY_plus_kernel, outX_plus_kernel];
                                    var result = kernel_value * output_value;
                                    str += $"+ {kernel_value}*{output_value}";

                                    // input coordinate from output coordinate equation

                                    // Store this result ... somewhere
                                    sum += result;
                                }
                            }

                            if (startX >= 0 && startX < input_width && startY >= 0 && startY < input_height)
                                input_error[startY, startX] += sum;
                            var mtx = Matrix<double>.Wrap(input_error).ToString();
                            Console.WriteLine(str);
                        }
                    }
                }

                batch_features[channelIndex] = Matrix<double>.Wrap(input_error);
            } 
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        }

        return new BatchedFeatureSet<double>(result_features);
    */

    // https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
    // https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
    // https://miro.medium.com/v2/resize:fit:720/format:webp/1*6iT86_FgzFIY7_fEoCrM_w.png
    private BatchedFeatureSet<double> TransposeConvolve2(ConvolutionLayer layer, BackpropagationArgs args) {
        // dx = dy_0 * w'
        var (batch_count, channel_count, input_height, input_width) = args.InputBatch.Shape;
        var filter_count = layer.FilterCount;
        var (_, _, output_height, output_width) = args.OutputBatch.Shape;
        var stride_x = layer.StrideX;
        var stride_y = layer.StrideY;

        var padding_rows = layer.RowsPadding;
        var padding_cols = layer.ColumnsPadding;

        var input_to_output_padding_rows = (input_height - output_height) / 2;
        var input_to_output_padding_cols = (input_width - output_width) / 2;

        var result_features = new FeatureSet<double>[batch_count];

        for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) {
            var batch_inputs = args.InputBatch[batchIndex];
            var batch_outputs = args.OutputBatch[batchIndex];
            var batch_errors = args.OutputErrors[batchIndex];

            var batch_features = new Matrix<double>[channel_count];
            for (var channelIndex = 0; channelIndex < channel_count; channelIndex++) {
                var input_error = new double[input_height, input_width];

                for (var filterIndex = 0; filterIndex < filter_count; filterIndex++) {
                    var filter = layer.Filters[filterIndex];
                    var filter_width = filter.Width;
                    var filter_height = filter.Height;

                    var filter_height_m1 = filter_height - 1;
                    var filter_width_m1 = filter_width - 1;

                    var input_padding_rows = (filter_height_m1) / 2;
                    var input_padding_cols = (filter_width_m1) / 2;

                    var input_width_padded = input_width + 2 * input_padding_rows;
                    var input_height_padded = input_height + 2 * input_padding_rows;
                    
                    var kernel = filter[channelIndex];
                    var error = batch_errors[filterIndex];
                    
                    // Slide kernel over input
                    for (var inputY = 0; inputY < input_height_padded; inputY++) {
                        var outY = (inputY - input_padding_rows - input_to_output_padding_rows) / stride_y;  // This assumes the output is "centered" in the middle of the input

                        for (var inputX = 0; inputX < input_width_padded; inputX++) {
                            var outX = (inputX - input_padding_cols - input_to_output_padding_cols) / stride_x; // This assumes the output is "centered" in the middle of the input

                            var sum = 0.0;
                            var str = "";
                            for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                                var inv_kernelY = filter_height_m1 - kernelY;
                                var outY_plus_kernel = outY + kernelY;
                                
                                for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                    var inv_kernelX = filter_width_m1 - kernelX;
                                    var outX_plus_kernel = outX + kernelX;

                                    var kernel_value = kernel[inv_kernelY, inv_kernelX];
                                    var output_value = error[outY_plus_kernel, outX_plus_kernel];
                                    var result = kernel_value * output_value;

                                    sum += result;      
                                    str += $"+ {kernel_value}*{output_value}";
                                }
                            }

                            if (inputX >= 0 && inputX < input_width && inputY >= 0 && inputY < input_height)
                                input_error[inputY, inputX] += sum;
                            var mtx = Matrix<double>.Wrap(input_error).ToString();
                            Console.WriteLine(str);
                        }
                    }
                }

                batch_features[channelIndex] = Matrix<double>.Wrap(input_error);
            } 
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        }

        return new BatchedFeatureSet<double>(result_features);
    }

    [TestMethod]
    public void TestTransposedConvolve() {
        var output_matrix = Matrix<double>.Wrap(new double[,]{
            {1, 3, 2, 1},
            {1, 3, 3, 1},
            {2, 1, 1, 3},
            {3, 2, 3, 3}
        });
        var output = new BatchedFeatureSet<double>(new FeatureSet<double>(output_matrix));

        var input_matrix = Matrix<double>.Wrap(new double[,] {
            {1, 5, 11, 14, 8, 3},
            {1, 6, 15, 18, 12, 3},
            {4, 13, 21, 21, 15, 11},
            {5, 17, 28, 27, 25, 11},
            {4, 7, 9, 12, 8, 6},
            {6, 7, 14, 13, 9, 6}
        });
        /*
        [
            1,5,11,14,8,3;
            1,6,15,18,12,3;
            4,13,21,21,15,11;
            5,17,28,27,25,11;
            4,7,9,12,8,6;
            6,7,14,13,9,6]
        */
        var input = new BatchedFeatureSet<double>(new FeatureSet<double>(input_matrix));

        var kernel = Matrix<double>.Wrap(new double[,] {
            {1, 2, 3},
            {0, 1, 0},
            {2, 1, 2}
        });
        var filters = new ConvolutionFilter(kernel);

        var temp_layer = new ConvolutionLayer(input_matrix.Shape, Padding.Same, filters);

        var results = TransposeConvolve2(temp_layer, new DotML.Network.Training.BatchTrainerEnumerator<FeedforwardNetwork>.BackpropagationArgs {
            InputBatch = input,
            OutputBatch = output,
            OutputErrors = output,
        });

        Assert.AreEqual(1, results.Batches);
        Assert.AreEqual(1, results.Channels);
        Assert.AreEqual(input.Rows, results.Rows);
        Assert.AreEqual(input.Columns, results.Columns);

        var result_matrix = results[0][0];
        Assert.AreEqual(input_matrix.Rows, result_matrix.Rows);
        Assert.AreEqual(input_matrix.Columns, result_matrix.Columns);
        foreach (var item in input_matrix.Zip(result_matrix)) {
            Assert.AreEqual(item.First, item.Second);
        }
    }

    private BatchedFeatureSet<double> DepthwiseTransposeConvolve2(DepthwiseConvolutionLayer layer, BackpropagationArgs args) {
        // dx = dy_0 * w'
        var (batch_count, channel_count, input_height, input_width) = args.InputBatch.Shape;
        var (_, _, output_height, output_width) = args.OutputBatch.Shape;
        var stride_x = layer.StrideX;
        var stride_y = layer.StrideY;

        var padding_rows = layer.RowsPadding;
        var padding_cols = layer.ColumnsPadding;

        var input_to_output_padding_rows = (input_height - output_height) / 2;
        var input_to_output_padding_cols = (input_width - output_width) / 2;

        var result_features = new FeatureSet<double>[batch_count];

        Parallel.For(0, batch_count, batchIndex => {
        //for (var batchIndex = 0; batchIndex < batch_count; batchIndex++) {
            var batch_inputs = args.InputBatch[batchIndex];
            var batch_outputs = args.OutputBatch[batchIndex];
            var batch_errors = args.OutputErrors[batchIndex];

            var batch_features = new Matrix<double>[channel_count];

            var filter = layer.Filter;
            var filter_width = filter.Width;
            var filter_height = filter.Height;

            var filter_height_m1 = filter_height - 1;
            var filter_width_m1 = filter_width - 1;

            var input_padding_rows = (filter_height_m1) / 2;
            var input_padding_cols = (filter_width_m1) / 2;

            var input_width_padded = input_width + 2 * input_padding_rows;
            var input_height_padded = input_height + 2 * input_padding_rows;

            for (var channelIndex = 0; channelIndex < channel_count; channelIndex++) {
                var input_error = new double[input_height, input_width];
                    
                var kernel = filter[channelIndex];
                var error = batch_errors[channelIndex];
                
                // Slide kernel over input
                for (var inputY = 0; inputY < input_height_padded; inputY++) {
                    var outY = (inputY - input_padding_rows - input_to_output_padding_rows) / stride_y;  // This assumes the output is "centered" in the middle of the input

                    for (var inputX = 0; inputX < input_width_padded; inputX++) {
                        var outX = (inputX - input_padding_cols - input_to_output_padding_cols) / stride_x; // This assumes the output is "centered" in the middle of the input

                        var sum = 0.0;
                        for (var kernelY = 0; kernelY < filter_height; kernelY++) {
                            var inv_kernelY = filter_height_m1 - kernelY;
                            var outY_plus_kernel = outY + kernelY;
                            
                            for (var kernelX = 0; kernelX < filter_width; kernelX++) {
                                var inv_kernelX = filter_width_m1 - kernelX;
                                var outX_plus_kernel = outX + kernelX;

                                var kernel_value = kernel[inv_kernelY, inv_kernelX];
                                var output_value = error[outY_plus_kernel, outX_plus_kernel];
                                var result = kernel_value * output_value;

                                sum += result;      
                            }
                        }

                        if (inputX >= 0 && inputX < input_width && inputY >= 0 && inputY < input_height)
                            input_error[inputY, inputX] += sum;
                    }
                }

                batch_features[channelIndex] = Matrix<double>.Wrap(input_error);
            } 
            result_features[batchIndex] = new FeatureSet<double>(batch_features);
        //}
        });

        return new BatchedFeatureSet<double>(result_features);
    }

    [TestMethod]
    public void TestTransposedConvolveDepthwise() {
        var output_matrix = Matrix<double>.Wrap(new double[,]{
            {1, 3, 2, 1},
            {1, 3, 3, 1},
            {2, 1, 1, 3},
            {3, 2, 3, 3}
        });
        var output = new BatchedFeatureSet<double>(new FeatureSet<double>(output_matrix));

        var input_matrix = Matrix<double>.Wrap(new double[,] {
            {1, 5, 11, 14, 8, 3},
            {1, 6, 15, 18, 12, 3},
            {4, 13, 21, 21, 15, 11},
            {5, 17, 28, 27, 25, 11},
            {4, 7, 9, 12, 8, 6},
            {6, 7, 14, 13, 9, 6}
        });
        /*
        [
            1,5,11,14,8,3;
            1,6,15,18,12,3;
            4,13,21,21,15,11;
            5,17,28,27,25,11;
            4,7,9,12,8,6;
            6,7,14,13,9,6]
        */
        var input = new BatchedFeatureSet<double>(new FeatureSet<double>(input_matrix));

        var kernel = Matrix<double>.Wrap(new double[,] {
            {1, 2, 3},
            {0, 1, 0},
            {2, 1, 2}
        });
        var filter = new ConvolutionFilter(kernel);

        var temp_layer = new DepthwiseConvolutionLayer(input_matrix.Shape, Padding.Same, filter);

        var results = DepthwiseTransposeConvolve2(temp_layer, new DotML.Network.Training.BatchTrainerEnumerator<FeedforwardNetwork>.BackpropagationArgs {
            InputBatch = input,
            OutputBatch = output,
            OutputErrors = output,
        });

        Assert.AreEqual(1, results.Batches);
        Assert.AreEqual(1, results.Channels);
        Assert.AreEqual(input.Rows, results.Rows);
        Assert.AreEqual(input.Columns, results.Columns);

        var result_matrix = results[0][0];
        Assert.AreEqual(input_matrix.Rows, result_matrix.Rows);
        Assert.AreEqual(input_matrix.Columns, result_matrix.Columns);
        foreach (var item in input_matrix.Zip(result_matrix)) {
            Assert.AreEqual(item.First, item.Second);
        }
    }
}