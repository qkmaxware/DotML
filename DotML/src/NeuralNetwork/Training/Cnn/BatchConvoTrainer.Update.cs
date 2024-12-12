using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

public partial class BatchedConvolutionalBackpropagationEnumerator<TNetwork> {

public struct LayerUpdateArgs {
    public int ParameterOffset;
    public int UpdateTimestep;
    public int LayerIndex;
    public Gradients? Gradients;
}
public struct LayerUpdateReturns {
    // Empty but left in case we need this in the future
}

private LayerUpdateActions layerUpdateActions {get; init;}
private class LayerUpdateActions: IConvolutionalLayerVisitor<BatchedConvolutionalBackpropagationEnumerator<TNetwork>.LayerUpdateArgs, BatchedConvolutionalBackpropagationEnumerator<TNetwork>.LayerUpdateReturns> {

    public double LearningRate {get; init;}
    public RegularizationFunction Regularization {get; init;}
    public ILearningRateOptimizer LearningRateOptimizer {get; init;}

    public LayerUpdateActions(double learningRate, RegularizationFunction regularization, ILearningRateOptimizer optimizer) {
        this.LearningRate = learningRate;
        this.Regularization = regularization;
        this.LearningRateOptimizer = optimizer;
    } 

    //private HashSet<int> used_params = new HashSet<int>();
    //if (used_params.Contains(parameterIndex))
            //throw new Exception("Bad parameter index " + parameterIndex);
        //used_params.Add(parameterIndex);
        //vf = beta * vi + (1.0 - beta) * grad;
        // W(t) = W(t-1) + alpha * vf;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double gradient_update(int updateTimestep, double learningRate, double prevWeight, double gradient, int parameterIndex) {
        var regularized_grad = gradient + Regularization.Invoke(prevWeight);
        var optimized_grad = LearningRateOptimizer.GetParameterUpdate(updateTimestep, learningRate, regularized_grad, parameterIndex);
        return optimized_grad;
    }

    public LayerUpdateReturns Visit(ConvolutionLayer layer, LayerUpdateArgs args) {
        if (args.Gradients is null || args.Gradients is not ConvolutionGradients gradients)
            throw new NullReferenceException(nameof(args.Gradients));

        var param_offset = args.ParameterOffset;        
        if (gradients.FilterKernelGradients is not null) {
            for (var filterIndex = 0; filterIndex < layer.FilterCount; filterIndex++) {
                var filter = layer.Filters[filterIndex];

                for (var kernelIndex = 0; kernelIndex < filter.Count; kernelIndex++) {
                    var kernel = filter[kernelIndex];
                    var kernel_width = kernel.Columns;
                    var grads = gradients.FilterKernelGradients[filterIndex][kernelIndex];
                    foreach (var elem in grads) {
                        if (double.IsNaN(elem)) {
                            throw new ArithmeticException($"NaN detected in kernel gradients for Filter {filterIndex}, Kernel {kernelIndex} while updating weights of a ConvolutionalLayer");
                        }
                    }
                    Matrix<double>.TransformInplace(grads, grads, (index, grad) => gradient_update(args.UpdateTimestep, LearningRate, kernel[index.Row, index.Column], grad, param_offset + index.Column + index.Row * kernel_width));
                    var kernel_update = grads;
                    Matrix<double>.SubInplace(grads, kernel, kernel_update);
                    var next_kernel = grads;
                    //var kernel_update = grads.Transform((index, grad) => gradient_update(args.UpdateTimestep, LearningRate, kernel[index.Row, index.Column], grad, param_offset + index.Column + index.Row * kernel_width));
                    //var next_kernel = kernel - kernel_update;
                    filter[kernelIndex] = next_kernel;
                    param_offset += kernel.Size;
                }
            }
        }

        if (gradients.BiasGradients is not null) {
            foreach (var elem in gradients.BiasGradients) {
                if (double.IsNaN(elem)) {
                    throw new ArithmeticException($"NaN detected in bias gradients while updating weights of a ConvolutionalLayer");
                }
            }
			for (var filterIndex = 0; filterIndex < layer.FilterCount; filterIndex++) {
				var filter = layer.Filters[filterIndex];
                var grads = gradients.BiasGradients[filterIndex];
                var bias_update = gradient_update(args.UpdateTimestep, LearningRate, filter.Bias, grads, param_offset + filterIndex);
				filter.Bias = filter.Bias - bias_update;
			}
		}

        return new LayerUpdateReturns {};
    }

    public LayerUpdateReturns Visit(PoolingLayer layer, LayerUpdateArgs args) {
        // Do nothing for gradient updates on the pooling layer
        return new LayerUpdateReturns {};
    }

    public LayerUpdateReturns Visit(DropoutLayer layer, LayerUpdateArgs args) {
        // Do nothing for gradient updates on the dropout layer
        return new LayerUpdateReturns {};
    }

    public LayerUpdateReturns Visit(FlatteningLayer layer, LayerUpdateArgs args) {
        // Do nothing for gradient updates on the flattening layer
        return new LayerUpdateReturns {};
    }

    public LayerUpdateReturns Visit(FullyConnectedLayer layer, LayerUpdateArgs args) {
        if (args.Gradients is null || args.Gradients is not FullyConnectedGradients gradients)
            throw new NullReferenceException(nameof(args.Gradients));

        var param_offset = args.ParameterOffset;
        var weight_count = layer.Weights.Size;
        var weight_width = layer.Weights.Columns;
        Matrix<double>.TransformInplace(gradients.WeightGradients, gradients.WeightGradients, (index, grad) => gradient_update(args.UpdateTimestep, LearningRate, layer.Weights[index.Row, index.Column], grad, param_offset + index.Column + index.Row * weight_width));
        var weight_update = gradients.WeightGradients;
        Matrix<double>.SubInplace(layer.Weights, layer.Weights, weight_update);
        //var weight_update = gradients.WeightGradients.Transform((index, grad) => gradient_update(args.UpdateTimestep, LearningRate, layer.Weights[index.Row, index.Column], grad, param_offset + index.Column + index.Row * weight_width));
        //layer.Weights = layer.Weights - weight_update;
        
        Vec<double>.TransformInplace(gradients.BiasGradients, gradients.BiasGradients, (index, grad) => gradient_update(args.UpdateTimestep, LearningRate, layer.Biases[index.Value], grad, param_offset + weight_count + index.Value));
        var bias_update = gradients.BiasGradients;
        Vec<double>.SubInplace(layer.Biases, layer.Biases, bias_update);
        //var bias_update = gradients.BiasGradients.Transform((index, grad) => gradient_update(args.UpdateTimestep, LearningRate, layer.Biases[index.Value], grad, param_offset + weight_count + index.Value));
        //layer.Biases = layer.Biases - bias_update;
        
        return new LayerUpdateReturns {};
    }

    public LayerUpdateReturns Visit(SoftmaxLayer layer, LayerUpdateArgs args) { 
        // Do nothing for gradient updates on the pooling layer
        return new LayerUpdateReturns {};
    }

}

}