using DotML.Network.Initialization;

namespace DotML.Network;

/// <summary>
/// Static class containing some common kernels/filters for ConvolutionLayer
/// </summary>
public static class Kernels {
    static HeInitialization he = new HeInitialization();
    /// <summary>
    /// Create a kernel of the given size with random weights from a He distribution
    /// </summary>
    /// <param name="size">Size of the kernel</param>
    /// <returns>matrix</returns>
    public static Matrix<float> HeKernel(int size) {
        var filter = new Matrix<float>(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                filter[i, j] = he.RandomWeight(size, size, size);
            }
        }
        return filter;
    }
    /// <summary>
    /// Create a kernel of the given size with random weights from a He distribution
    /// </summary>
    /// <param name="rows">Rows in the kernel</param>
    /// <param name="columns">Columns in the kernel</param>
    /// <returns>matrix</returns>
    public static Matrix<float> HeKernel(int rows, int columns) {
        var filter = new Matrix<float>(rows, columns);
        var size = filter.Size;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                filter[i, j] = he.RandomWeight(size, size, size);
            }
        }
        return filter;
    }
    static NormalXavierInitialization xavier = new NormalXavierInitialization();
    /// <summary>
    /// Create a kernel of the given size with random weights from a normal Xavier distribution
    /// </summary>
    /// <param name="size">Size of the kernel</param>
    /// <returns>matrix</returns>
    public static Matrix<float> XavierKernel(int size) {
        var filter = new Matrix<float>(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                filter[i, j] = xavier.RandomWeight(size, size, size);
            }
        }
        return filter;
    }
    /// <summary>
    /// Create a kernel of the given size with random weights from a normal Xavier distribution
    /// </summary>
    /// <param name="rows">Rows in the kernel</param>
    /// <param name="columns">Columns in the kernel</param>
    /// <returns>matrix</returns>
    public static Matrix<float> XavierKernel(int rows, int columns) {
        var filter = new Matrix<float>(rows, columns);
        var size = filter.Size;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                filter[i, j] = xavier.RandomWeight(size, size, size);
            }
        }
        return filter;
    }

    static Random rand = new Random();
    /// <summary>
    /// Create a kernel of the given size with random weights
    /// </summary>
    /// <param name="size">Size of the kernel</param>
    /// <returns>matrix</returns>
    public static Matrix<float> RandomKernel(int size) {
        size = Math.Max(1, size);
        var filter = new Matrix<float>(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
            {
                filter[i, j] = (float)(rand.NextDouble() * 2 - 1); // Random weights
            }
        }
        return filter;
    }

    /// <summary>
    /// Identity kernel
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> IdentityKernel(int size) {
        return Matrix<float>.Identity(size);
    }

    /// <summary>
    /// Averages the pixel values in the kernel area, resulting in a blur effect
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> BoxBlurKernel(int size) {
        const float oneNinth = 1.0f/9.0f;
        return new Matrix<float>(new float[,]{
            { oneNinth, oneNinth, oneNinth },
            { oneNinth, oneNinth, oneNinth },
            { oneNinth, oneNinth, oneNinth },
        });
    }

    /// <summary>
    /// A weighted average blur that reduces image noise and detail, based on a Gaussian function
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> GaussianBlurKernel(int size) {
        const float oneSixteenth = 1.0f/16.0f;
        const float twoSixteenth = 2.0f/16.0f;
        const float fourSixteenth = 4.0f/16.0f;
        return new Matrix<float>(new float[,]{
            { oneSixteenth, twoSixteenth, oneSixteenth },
            { twoSixteenth, fourSixteenth, twoSixteenth },
            { oneSixteenth, twoSixteenth, oneSixteenth },
        });
    }

    /// <summary>
    /// Sobel kernel for edge detection, particularly for finding horizontal edges
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> SobelXKernel() {
        return new Matrix<float>(new float[,]{
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        });
    }

    /// <summary>
    /// Sobel kernel for edge detection, particularly for finding horizontal edges
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> SobelYKernel() {
        return new Matrix<float>(new float[,]{
            { -1, -2, -1 },
            { 0, 0, 0 },
            { 1, 2, 1 }
        });
    }

    /// <summary>
    /// Prewitt kernel for edge detection, similar to Sobel but with different weights
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> PrewittXKernel() {
        return new Matrix<float>(new float[,]{
            { -1, 0, 1 },
            { -1, 0, 1 },
            { -1, 0, 1 }
        });
    }

    /// <summary>
    /// Prewitt kernel for edge detection, similar to Sobel but with different weights
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> PrewittYKernel() {
        return new Matrix<float>(new float[,]{
            { -1, -1, -1 },
            { 0, 0, 0 },
            { 1, 1, 1 }
        });
    }

    /// <summary>
    /// Detects areas of rapid intensity change, commonly used for edge detection and sharpening
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> LaplacianKernel() {
        return new Matrix<float>(new float[,]{
            { 0, 1, 0 },
            { 1, -4, 1 },
            { 0, 1, 0 }
        });
    }

    /// <summary>
    /// Enhances the edges and fine details of an image by emphasizing differences between neighboring pixels
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> SharpeningKernel() {
        return new Matrix<float>(new float[,]{
            { 0, -1, 0 },
            {-1, 5, -1 },
            { 0, -1, 0 }
        });
    }

    /// <summary>
    /// Gives the image a 3D shadow effect, emphasizing edges and contours
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<float> EmbossKernel() {
        return new Matrix<float>(new float[,]{
            { -2, -1, 0 },
            {-1, 1, 1 },
            { 0, 1, 2 }
        });
    }
}