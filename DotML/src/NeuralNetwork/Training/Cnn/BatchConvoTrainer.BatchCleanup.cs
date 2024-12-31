using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotML.Network.Initialization;

namespace DotML.Network.Training;

public partial class BatchedConvolutionalBackpropagationEnumerator<TNetwork> {

    private readonly BatchCleanup batchCleanup = new BatchCleanup();
    private class BatchCleanup : IConvolutionalLayerVisitor {
        public void Visit(ConvolutionLayer layer) { }

        public void Visit(DepthwiseConvolutionLayer layer) { }

        public void Visit(PoolingLayer layer) { }

        public void Visit(FlatteningLayer layer) { }

        public void Visit(DropoutLayer layer) {
            // Stop using a shared mask
            layer.UseSharedMask = false;
            layer.ClearSharedMask();
        }

        public void Visit(LayerNorm layer) {}

        public void Visit(BatchNorm layer) {}

        public void Visit(FullyConnectedLayer layer) { }

        public void Visit(ActivationLayer layer) { }

        public void Visit(SoftmaxLayer layer) { }
    }
} 