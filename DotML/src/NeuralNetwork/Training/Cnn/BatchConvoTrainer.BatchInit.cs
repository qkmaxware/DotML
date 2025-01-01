using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

public partial class BatchedConvolutionalBackpropagationEnumerator<TNetwork> {

    private readonly BatchInitializer batchInitializer = new BatchInitializer();
    private class BatchInitializer : IConvolutionalLayerVisitor {
        public void Visit(ConvolutionLayer layer) { }

        public void Visit(DepthwiseConvolutionLayer layer) { }

        public void Visit(PoolingLayer layer) { }

        public void Visit(FlatteningLayer layer) { }

        public void Visit(DropoutLayer layer) {
            // Tell the layer to share the same mask across each batch item. Generate a new mask for this batch.
            layer.UseSharedMask = true;
            layer.ClearSharedMask();
        }

        public void Visit(LayerNorm layer) {}

        public void Visit(BatchNorm layer) {
            layer.IsTrainingMode = true;
        }

        public void Visit(FullyConnectedLayer layer) { }

        public void Visit(ActivationLayer layer) { }

        public void Visit(SoftmaxLayer layer) { }
    }
} 