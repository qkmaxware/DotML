namespace System
{
    /// <summary>
    /// Value type representing nothing
    /// </summary>
    public readonly struct None
    {
        /// <summary>
        /// The only possible value for None
        /// </summary>
        public static readonly None Value = new None();

        public override string ToString() => "None";
    }
}

namespace DotML.Network
{
    /// <summary>
    /// Interface for blocks that can be visited by a block visitor
    /// </summary>
    public interface IBlockVisitable
    {
        /// <summary>
        /// Accepts a visitor and dispatches the appropriate Visit method based on the current block type.
        /// </summary>
        /// <typeparam name="TArg">additional argument type</typeparam>
        /// <typeparam name="TResult">visit return type</typeparam>
        /// <param name="arg">additional argument</param>
        /// <param name="visitor">visitor</param>
        /// <returns>visitor return</returns>
        public TResult Accept<TArg, TResult>(IBlockVisitor<TArg, TResult> visitor, TArg arg);
    }
    
    /// <summary>
    /// Utility helper method for default implementations of Accept for no-arg/no-return visitors
    /// </summary>
    public static class BlockVisitableExtensions
    {
        public static TResult Accept<TResult>(
            this IBlockVisitable visitable,
            IBlockVisitor<TResult> visitor)
        {
            return visitable.Accept(visitor, None.Value);
        }

        public static void Accept(
            this IBlockVisitable visitable,
            IBlockVisitor visitor)
        {
            visitable.Accept(visitor, None.Value);
        }
    }

    /// <summary>
    /// Interface for visitor pattern for blocks
    /// </summary>
    /// <typeparam name="TArg">additional argument type</typeparam>
    /// <typeparam name="TResult">visit result type</typeparam>
    public interface IBlockVisitor<TArg, TResult>
    {
        /// <summary>
        /// A default fallback visit method for unhandled types
        /// </summary>
        /// <exception cref="NotSupportedException">always</exception>
        public TResult Visit(object? obj, TArg arg) => throw new NotSupportedException();

        #region Standalone Layers
        public TResult Visit(Activation activation, TArg arg);
        public TResult Visit(SoftmaxOutput softmax, TArg arg);
        
        public TResult Visit(Conv2D conv, TArg arg);
        public TResult Visit(TransposeConv2D tconv, TArg arg);

        public TResult Visit(DenseLinear dense, TArg arg);

        public TResult Visit(Dropout dropout, TArg arg);

        public TResult Visit(BatchNorm2D norm, TArg arg);
        public TResult Visit(GroupNorm norm, TArg arg);
        public TResult Visit(LayerNorm norm, TArg arg);

        public TResult Visit(PixelShuffler shuffle, TArg arg);

        public TResult Visit(AvgPool2D pool, TArg arg);
        public TResult Visit(MaxPool2D pool, TArg arg);
        public TResult Visit(MinPool2D pool, TArg arg);

        public TResult Visit(GlobalAvgPool2D pool, TArg arg);
        public TResult Visit(GlobalMaxPool2D pool, TArg arg);
        public TResult Visit(GlobalMinPool2D pool, TArg arg);

        public TResult Visit(Reshape reshape, TArg arg);
        public TResult Visit(Flatten flatten, TArg arg);

        public TResult Visit(Center2D center, TArg arg);
        #endregion

        #region Structural Blocks
        public TResult Visit(ResidualBlock block, TArg arg);
        public TResult Visit(SequentialBlock block, TArg arg);
        #endregion
    }

    /// <summary>
    /// A block visitor that doesn't have any arguments but has a return value
    /// </summary>
    /// <typeparam name="TResult">visit result type</typeparam>
    public interface IBlockVisitor<TResult> : IBlockVisitor<None, TResult> { }

    /// <summary>
    /// A block visitor that doesn't have any arguments or return value
    /// </summary>
    public interface IBlockVisitor : IBlockVisitor<None, None> { }
        
}