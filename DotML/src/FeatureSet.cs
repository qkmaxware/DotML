using System.Collections;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;

namespace DotML;

/// <summary>
/// Represents a batch of multiple feature sets
/// </summary>
/// <typeparam name="T">feature value type</typeparam>
public class BatchedFeatureSet<T> :
    BaseFeatureTensor<T>,
    IEnumerable<FeatureSet<T>>,
    IMutableTensorLike<T>
    where T:INumber<T>
{
    private FeatureSet<T>[] batches;

    /// <summary>
    /// Number of batches of feature sets
    /// </summary>
    public int Batches {get; init;}

    /// <summary>
    /// Number of channels in the feature set
    /// </summary>
    public int Channels {get; init;}

    /// <summary>
    /// Number of rows in the feature set
    /// </summary>
    public int Rows {get; init;}

    /// <summary>
    /// Number of columns in the feature set
    /// </summary>
    public int Columns {get; init;}

    /// <summary>
    /// Shape of the feature set
    /// </summary>
    public Shape4D Shape => new Shape4D(Batches, Channels, Rows, Columns);

    /// <summary>
    /// Number of dimensions in this tensor
    /// </summary>
    public override int Rank => 4;

    /// <summary>
    /// Create a new empty batched feature set
    /// </summary>
    public BatchedFeatureSet() {
        batches = Array.Empty<FeatureSet<T>>();
        Batches = 0;
        Channels = 0;
        Rows = 0;
        Columns = 0;
    }

    /// <summary>
    /// Copy an existing batched feature set
    /// </summary>
    /// <param name="other"></param>
    public BatchedFeatureSet(BatchedFeatureSet<T> other) {
        this.batches = (FeatureSet<T>[])other.batches.Clone();
        Batches = other.Batches;
        Channels = other.Channels;
        Rows = other.Rows;
        Columns = other.Columns;
    }

    /// <summary>
    /// Create a new batched feature set from the list of feature sets
    /// </summary>
    /// <param name="batches">feature sets</param>
    public BatchedFeatureSet(params FeatureSet<T>[] batches) {
        this.batches    = batches;
        Batches         = batches.Length;
        Channels        = batches.Length > 0 ? batches[0].Channels : 0;
        Rows            = batches.Length > 0 ? batches[0].Rows : 0;
        Columns         = batches.Length > 0 ? batches[0].Columns : 0;
    }

    /// <summary>
    /// Create a batched feature set with the given shape
    /// </summary>
    /// <param name="shape">shape of the feature set</param>
    public BatchedFeatureSet(Shape4D shape) {
        var batches = new FeatureSet<T>[shape.Batches];
        var subshape = new Shape3D(shape.Channels, shape.Rows, shape.Columns);
        for (var batch = 0; batch < shape.Batches; batch++) {
            batches[batch] = new FeatureSet<T>(subshape);
        }
        this.batches    = batches;
        Batches         = shape.Batches;
        Channels        = shape.Channels;
        Rows            = shape.Rows;
        Columns         = shape.Columns;
    }

    /// <summary>
    /// Create a batched feature set from a rectangular array with the first dimension being the number of batches, then features/channels and lastly the rows and columns
    /// </summary>
    /// <param name="tensor">4d tensor as a rectangular array</param>
    /// <returns>batched feature set</returns>
    public static BatchedFeatureSet<T> FromRectangular(T[,,,] tensor) {
        var batch_count = tensor.GetLength(0);
        var channel_count = tensor.GetLength(1);
        var rows = tensor.GetLength(2);
        var cols = tensor.GetLength(3);
        var batches = new FeatureSet<T>[batch_count];

        for (var b = 0; b < batch_count; b++) {
            var channels = new Matrix<T>[channel_count];
            for (var c = 0; c < channel_count; c++) {
                var channel = new Matrix<T>(rows, cols);
                for (var row = 0; row < rows; row++) {
                    for (var col = 0; col < cols; col++) {
                        channel[row, col] = tensor[b, c, row, col];
                    }
                }
                channels[c] = channel;
            }
            batches[b] = new FeatureSet<T>(channels);
        }
        return new BatchedFeatureSet<T>(batches);
    }

    /// <summary>
    /// Create a batched feature set from a jagged array with the first dimension being the number of batches, then features/channels and lastly the rows and columns
    /// </summary>
    /// <param name="tensor">4d tensor as a jagged array</param>
    /// <returns>batched feature set</returns>
    public static BatchedFeatureSet<T> FromJagged(T[][][][] tensor) {
        var batch_count = tensor.Length;
        var batches = new FeatureSet<T>[batch_count];
        for (var b = 0; b < batch_count; b++) {
            batches[b] = FeatureSet<T>.FromJagged(tensor[b]);
        }
        return new BatchedFeatureSet<T>(batches);
    }

    /// <summary>
    /// Fetch a given feature by channel index
    /// </summary>
    /// <param name="batch">batch index</param>
    /// <returns>feature set</returns>
    public FeatureSet<T> this[int batch] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => batches[batch];
    }

    /// <summary>
    /// Fetch a feature matrix
    /// </summary>
    /// <param name="batch">batch index</param>
    /// <param name="channel">channel index</param>
    /// <returns>feature</returns>
    public Matrix<T> this[int batch, int channel] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => batches[batch][channel];
    }

    /// <summary>
    /// Fetch a feature value
    /// </summary>
    /// <param name="batch">batch index</param>
    /// <param name="channel">channel index</param>
    /// <param name="row">row index</param>
    /// <param name="col">column index</param>
    /// <returns>feature</returns>
    public T this[int batch, int channel, int row, int col] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => batches[batch][channel][row, col];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            var m = batches[batch][channel];
            m[row, col] = value;
        }
    }

    /// <summary>
    /// Get the length of the given tensor dimension by index
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>Dimension length</returns>
    public override int GetDimension(int index) {
        return index switch {
            0 => Batches,
            1 => Channels,
            2 => Rows,
            3 => Columns,
            _ => 1
        };
    }

    /// <summary>
    /// Get the element at the given index
    /// </summary>
    /// <param name="indices">Set of indices across all 4 dimensions</param>
    /// <returns>Element</returns>
    public override T GetElementAt(params int[] indices) {
        return this[indices[0],indices[1],indices[2],indices[3]];
    }

    /// <summary>
    /// Set the element at the given index
    /// </summary>
    /// <param name="value">Value to set</param>
    /// <param name="indices">Set of indices across all 4 dimensions</param>
    public override void SetElementAt(T value, params int[] indices) {
        this[indices[0],indices[1],indices[2],indices[3]] = value;
    }

    public IEnumerator<FeatureSet<T>> GetEnumerator() => ((IEnumerable<FeatureSet<T>>)batches).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => batches.GetEnumerator();

    /// <summary>
    /// Fetch the underlying feature set array
    /// </summary>
    /// <returns>feature set array</returns>
    public FeatureSet<T>[] AsArray() => this.batches;

    public string ToJaggedArrayString() {
        StringBuilder sb = new StringBuilder();
        sb.Append('[');
        for (var batch = 0; batch < this.Batches; batch++) {
            if (batch != 0)
                sb.Append(',');
            sb.Append(this[batch].ToJaggedArrayString());
        }
        sb.Append(']');
        return sb.ToString();
    }
}

/// <summary>
/// Feature set (basically a wrapper for a matrix array with additional semantics)
/// </summary>
/// <typeparam name="T">feature value type</typeparam>
public class FeatureSet<T> :
    BaseFeatureTensor<T>,
    IEnumerable<Matrix<T>>,
    IMutableTensorLike<T>
    where T:INumber<T>
{
    private Matrix<T>[] channels; // The feature channels

    #region Constructors

    /// <summary>
    /// Create an empty feature set
    /// </summary>
    public FeatureSet() {
        this.channels = Array.Empty<Matrix<T>>();
        this.Channels = 0;
        this.Rows = 0;
        this.Columns = 0;
    }

    /// <summary>
    /// Create a copy of a feature set
    /// </summary>
    /// <param name="other">feature set to copy</param>
    public FeatureSet(FeatureSet<T> other) {
        this.channels = (Matrix<T>[])other.channels.Clone();
        this.Channels = other.Channels;
        this.Rows = other.Rows;
        this.Columns = other.Columns;
    }

    /// <summary>
    /// Create a feature set from the given channels
    /// </summary>
    /// <param name="channels">feature channels</param>
    public FeatureSet(params Matrix<T>[] channels) {
        this.channels   = channels;
        this.Channels   = channels.Length;
        this.Rows       = channels.Length > 0 ? channels[0].Rows : 0;
        this.Columns    = channels.Length > 0 ? channels[0].Columns : 0;
    }

    /// <summary>
    /// Create a feature set with the given shape
    /// </summary>
    /// <param name="channels">shape of the feature set</param>
    public FeatureSet(Shape3D shape) {
        var channels = new Matrix<T>[shape.Channels];
        for (var c = 0; c < shape.Channels; c++) {
            channels[c] = new Matrix<T>(shape.Rows, shape.Columns);
        }
        this.channels   = channels;
        this.Channels   = channels.Length;
        this.Rows       = channels.Length > 0 ? channels[0].Rows : 0;
        this.Columns    = channels.Length > 0 ? channels[0].Columns : 0;
    }

    /// <summary>
    /// Create a feature set from a rectangular array with the first dimension being the number of features/channels and the next 2 being the rows and columns
    /// </summary>
    /// <param name="tensor">3d tensor as a rectangular array</param>
    /// <returns>feature set</returns>
    public static FeatureSet<T> FromRectangular(T[,,] tensor) {
        var channel_count = tensor.GetLength(0);
        var rows = tensor.GetLength(1);
        var cols = tensor.GetLength(2);
        var channels = new Matrix<T>[channel_count];
        for (var c = 0; c < channel_count; c++) {
            var channel = new Matrix<T>(rows, cols);
            for (var row = 0; row < rows; row++) {
                for (var col = 0; col < cols; col++) {
                    channel[row, col] = tensor[c, row, col];
                }
            }
            channels[c] = channel;
        }
        return new FeatureSet<T>(channels);
    }

    /// <summary>
    /// Create a feature set from a jagged array with the first dimension being the number of features/channels and the next 2 being the rows and columns
    /// </summary>
    /// <param name="tensor">3d tensor as a jagged array</param>
    /// <returns>feature set</returns>
    public static FeatureSet<T> FromJagged(T[][][] tensor) {
        var channel_count = tensor.Length;
        var channels = new Matrix<T>[channel_count];
        for (var c = 0; c < channel_count; c++) {
            channels[c] = Matrix<T>.FromJagged(tensor[c]);
        }
        return new FeatureSet<T>(channels);
    }

    #endregion

    #region Shape

    /// <summary>
    /// Number of channels in the feature set
    /// </summary>
    public int Channels {get; init;}

    /// <summary>
    /// Number of rows in the feature set
    /// </summary>
    public int Rows {get; init;}

    /// <summary>
    /// Number of columns in the feature set
    /// </summary>
    public int Columns {get; init;}

    /// <summary>
    /// Shape of the feature set
    /// </summary>
    public Shape3D Shape => new Shape3D(Channels, Rows, Columns);

    /// <summary>
    /// Number of dimensions in this tensor
    /// </summary>
    public override int Rank => 3;

    /// <summary>
    /// Get the length of the given tensor dimension by index
    /// </summary>
    /// <param name="index">dimension index</param>
    /// <returns>Dimension length</returns>
    public override int GetDimension(int index) {
        return index switch {
            0 => Channels,
            1 => Rows,
            2 => Columns,
            _ => 1
        };
    }

    /// <summary>
    /// Get the element at the given index
    /// </summary>
    /// <param name="indices">Set of indices across all 4 dimensions</param>
    /// <returns>Element</returns>
    public override T GetElementAt(params int[] indices) {
        return this[indices[0],indices[1],indices[2]];
    }

    /// <summary>
    /// Set the element at the given index
    /// </summary>
    /// <param name="value">Value to set</param>
    /// <param name="indices">Set of indices across all 4 dimensions</param>
    public override void SetElementAt(T value, params int[] indices) {
        this[indices[0],indices[1],indices[2]] = value;
    }

    #endregion

    #region Indexers

    /// <summary>
    /// Fetch a given feature by channel index
    /// </summary>
    /// <param name="channel">channel index</param>
    /// <returns>feature matrix</returns>
    public Matrix<T> this[int channel] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => channels[channel];
    }

    /// <summary>
    /// Fetch a feature value
    /// </summary>
    /// <param name="channel">channel index</param>
    /// <param name="row">row index</param>
    /// <param name="col">column index</param>
    /// <returns>feature value</returns>
    public T this[int channel, int row, int col] {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => channels[channel][row, col];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set {
            var m = channels[channel];
            m[row, col] = value;
        }
    }

    #endregion

    #region Channel Shortcuts

    /// <summary>
    /// The X channel (channel 0)
    /// </summary>
    public Matrix<T> X => channels.ElementAt(0);
    /// <summary>
    /// The Y channel (channel 1)
    /// </summary>
    public Matrix<T> Y => channels.ElementAt(1);
    /// <summary>
    /// The Z channel (channel 2)
    /// </summary>
    public Matrix<T> Z => channels.ElementAt(2);

    /// <summary>
    /// The Red image channel (channel 0)
    /// </summary>
    public Matrix<T> Red => channels.ElementAt(0);
    /// <summary>
    /// The Green image channel (channel 1)
    /// </summary>
    public Matrix<T> Green => channels.ElementAt(1);
    /// <summary>
    /// The Blue image channel (channel 2)
    /// </summary>
    public Matrix<T> Blue => channels.ElementAt(2);
    /// <summary>
    /// The Alpha image channel (channel 3)
    /// </summary>
    public Matrix<T> Alpha => channels.ElementAt(3);

    #endregion

    /// <summary>
    /// Concatenate additional feature sets to create a new feature set
    /// </summary>
    /// <param name="other">feature set to concatenate</param>
    /// <returns>new feature set containing the features of both</returns>
    public FeatureSet<T> Concat(FeatureSet<T> other) {
        Matrix<T>[] results = new Matrix<T>[this.channels.Length + other.channels.Length];
        this.channels.CopyTo(results, 0);
        other.channels.CopyTo(results, this.channels.Length);
        return new FeatureSet<T>(results);
    }

    public IEnumerator<Matrix<T>> GetEnumerator() => ((IEnumerable<Matrix<T>>)channels).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => channels.GetEnumerator();

    /// <summary>
    /// Convert a matrix array to feature set
    /// </summary>
    /// <param name="channels">array of feature channels</param>
    public static explicit operator FeatureSet<T>(Matrix<T>[] channels) => new FeatureSet<T>(channels);

    /// <summary>
    /// Convert a feature set to a matrix array
    /// </summary>
    /// <param name="features">feature set</param>
    public static explicit operator Matrix<T>[](FeatureSet<T> features) => features.channels;

    /// <summary>
    /// Fetch the underlying matrix array interpretation of the data
    /// </summary>
    /// <returns>matrix array</returns>
    public Matrix<T>[] AsArray() => this.channels;

    public string ToJaggedArrayString() {
        StringBuilder sb = new StringBuilder();
        sb.Append('[');
        for (var feature = 0; feature < this.Channels; feature++) {
            if (feature != 0)
                sb.Append(',');
            sb.Append(this[feature].ToJaggedArrayString());
        }
        sb.Append(']');
        return sb.ToString();
    }
}

/// <summary>
/// Base class with multi-dimensional tensor operators for tensors representing neural network input/output features
/// </summary>
/// <typeparam name="T">feature value type</typeparam>
public abstract class BaseFeatureTensor<T> : IMutableTensorLike<T> where T:INumber<T> {
    #region Implement IMutableTensorLike
    public abstract int Rank { get; }

    public abstract int GetDimension(int index);

    public abstract T GetElementAt(params int[] indices);

    public abstract void SetElementAt(T value, params int[] indices);
    #endregion

    /// <summary>
    /// Size of the tensor (number of elements)
    /// </summary>
    public int Size => Enumerable.Range(0, Rank).Aggregate(1, (old, diff) => old * GetDimension(diff));

    /// <summary>
    /// Enumerate over the elements of the features tensor in row-major order
    /// </summary>
    /// <returns>Enumerable of items</returns>
    public IEnumerable<T> FlattenElements() {
        int[] indices = new int[Rank];
        while (true) {
            // Return the combination
            yield return GetElementAt(indices);

            // Find the rightmost dimension to increment
            int i = Rank - 1;
            while (i >= 0) {
                indices[i]++;
                if (indices[i] < GetDimension(i)) // If the index is within bounds
                {
                    break;
                }
                indices[i] = 0; // Reset to 0 if out of bounds and continue incrementing left
                i--;
            }

            // If all indices have been exhausted (i.e., we've generated all combinations), break the loop
            if (i < 0)
                break;
        }
    }

    private static T NextElement(IEnumerator<T> enumerator) {
        if (enumerator.MoveNext()) {
            return enumerator.Current;
        } else {
            return T.Zero;
        }
    }

    /// <summary>
    /// Permutate this batched feature set into another
    /// </summary>
    /// <param name="dim1">dimension to use for the first dimension (batches)</param>
    /// <param name="dim2">dimension to use for the second dimension (channels)</param>
    /// <param name="dim3">dimension to use for the third dimension (rows)</param>
    /// <param name="dim4">dimension to use for the fourth dimension (columns)</param>
    /// <returns>Permuted batched feature set</returns>
    public BatchedFeatureSet<T> Permute(int dim1, int dim2, int dim3, int dim4) {
        dim1 = Math.Clamp(dim1, 0, Rank);
        dim2 = Math.Clamp(dim2, 0, Rank);
        dim3 = Math.Clamp(dim3, 0, Rank);
        dim4 = Math.Clamp(dim4, 0, Rank);

        var next = new BatchedFeatureSet<T>(new Shape4D(GetDimension(dim1), GetDimension(dim2), GetDimension(dim3), GetDimension(dim4)));
        var dim_mapping = new int[Math.Max(4, Rank)];
        for (var i = 0; i < next.Batches; i++) {
            dim_mapping[dim1] = i;
            var batch = next[i];
            for (var f = 0; f < next.Channels; f++) {
                dim_mapping[dim2] = f;
                var feat = batch[f];
                for (var r = 0; r < next.Rows; r++) {
                    dim_mapping[dim3] = r;
                    for (var c = 0; c < next.Columns; c++) {
                        dim_mapping[dim4] = c;
                        feat[r, c] = GetElementAt(dim_mapping);
                    }
                }
            }
        }
        return next;
    }

    /// <summary>
    /// Permutate this feature set into another
    /// </summary>
    /// <param name="dim1">dimension to use for the second dimension (channels)</param>
    /// <param name="dim2">dimension to use for the third dimension (rows)</param>
    /// <param name="dim3">dimension to use for the fourth dimension (columns)</param>
    /// <returns>Permuted feature set</returns>
    public FeatureSet<T> Permute(int dim1, int dim2, int dim3) {
        dim1 = Math.Clamp(dim1, 0, Rank);
        dim2 = Math.Clamp(dim2, 0, Rank);
        dim3 = Math.Clamp(dim3, 0, Rank);

        var next = new FeatureSet<T>(new Shape3D(GetDimension(dim1), GetDimension(dim2), GetDimension(dim3)));
        var dim_mapping = new int[Math.Max(3, Rank)];

        for (var f = 0; f < next.Channels; f++) {
            dim_mapping[dim1] = f;
            var feat = next[f];
            for (var r = 0; r < next.Rows; r++) {
                dim_mapping[dim2] = r;
                for (var c = 0; c < next.Columns; c++) {
                    dim_mapping[dim3] = c;
                    feat[r, c] = GetElementAt(dim_mapping);
                }
            }
        }
        return next;
    }

    /// <summary>
    /// Permutate this feature into another
    /// </summary>
    /// <param name="dim1">dimension to use for the second dimension (rows)</param>
    /// <param name="dim2">dimension to use for the third dimension (columns)</param>
    /// <returns>Permuted feature</returns>
    public Matrix<T> Permute(int dim1, int dim2) {
        dim1 = Math.Clamp(dim1, 0, Rank);
        dim2 = Math.Clamp(dim2, 0, Rank);

        var next = new Matrix<T>(new Shape2D(GetDimension(dim1), GetDimension(dim2)));
        var dim_mapping = new int[Math.Max(2, Rank)];


        for (var r = 0; r < next.Rows; r++) {
            dim_mapping[dim1] = r;
            for (var c = 0; c < next.Columns; c++) {
                dim_mapping[dim2] = c;
                next[r, c] = GetElementAt(dim_mapping);
            }
        }
        return next;
    }

    /// <summary>
    /// Reshape this tensor into another 4d shape
    /// </summary>
    /// <param name="shape">Shape to reshape into</param>
    /// <returns>Newly shaped tensor</returns>
    public BatchedFeatureSet<T> Reshape(Shape4D shape) {
        var batches = new BatchedFeatureSet<T>(shape);

        var item_generator = FlattenElements().GetEnumerator();

        foreach (var batch in batches) {
            foreach (var feature in batch) {
                var matrix = feature;
                for (var row = 0; row < matrix.Rows; row++) {
                    for (var col = 0; col < matrix.Columns; col++) {
                        matrix[row, col] = NextElement(item_generator);
                    }
                }
            }
        }

        return batches;
    }

    /// <summary>
    /// Reshape this tensor into another 3d shape
    /// </summary>
    /// <param name="shape">Shape to reshape into</param>
    /// <returns>Newly shaped tensor</returns>
    public FeatureSet<T> Reshape(Shape3D shape) {
        var features = new FeatureSet<T>(shape);

        var item_generator = FlattenElements().GetEnumerator();

        foreach (var feature in features) {
            var matrix = feature;
            for (var row = 0; row < matrix.Rows; row++) {
                for (var col = 0; col < matrix.Columns; col++) {
                    matrix[row, col] = NextElement(item_generator);
                }
            }
        }

        return features;
    }

    /// <summary>
    /// Reshape this tensor into a single 2d matrix
    /// </summary>
    /// <param name="shape">Shape to reshape into</param>
    /// <returns>Newly shaped tensor</returns>
    public Matrix<T> Reshape(Shape2D shape) {
        Matrix<T> matrix = new Matrix<T>(shape);
        var item_generator = FlattenElements().GetEnumerator();
        for (var row = 0; row < matrix.Rows; row++) {
            for (var col = 0; col < matrix.Columns; col++) {
                matrix[row, col] = NextElement(item_generator);
            }
        }
        return matrix;
    }

    /// <summary>
    /// Convert the feature set matrices to a single flattened vector
    /// </summary>
    /// <returns></returns>
    public Vec<T> ToVector() => Vec<T>.Wrap(FlattenElements().ToArray());
}