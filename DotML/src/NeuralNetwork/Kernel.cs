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
    public static Matrix<double> HeKernel(int size) {
        double[,] filter = new double[size, size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                filter[i, j] = he.RandomWeight(size, size, size);
            }
        }
        return Matrix<double>.Wrap(filter);
    }
    static NormalXavierInitialization xavier = new NormalXavierInitialization();
    /// <summary>
    /// Create a kernel of the given size with random weights from a normal Xavier distribution
    /// </summary>
    /// <param name="size">Size of the kernel</param>
    /// <returns>matrix</returns>
    public static Matrix<double> XavierKernel(int size) {
        double[,] filter = new double[size, size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                filter[i, j] = xavier.RandomWeight(size, size, size);
            }
        }
        return Matrix<double>.Wrap(filter);
    }

    static Random rand = new Random();
    /// <summary>
    /// Create a kernel of the given size with random weights
    /// </summary>
    /// <param name="size">Size of the kernel</param>
    /// <returns>matrix</returns>
    public static Matrix<double> RandomKernel(int size) {
        size = Math.Max(1, size);
        double[,] filter = new double[size, size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
            {
                filter[i, j] = (float)(rand.NextDouble() * 2 - 1); // Random weights
            }
        }
        return Matrix<double>.Wrap(filter);
    }

    /// <summary>
    /// Identity kernel
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> IdentityKernel(int size) {
        return Matrix<double>.Identity(size);
    }

    /// <summary>
    /// Averages the pixel values in the kernel area, resulting in a blur effect
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> BoxBlurKernel(int size) {
        const double oneNinth = 1.0/9.0;
        return new Matrix<double>(new double[,]{
            { oneNinth, oneNinth, oneNinth },
            { oneNinth, oneNinth, oneNinth },
            { oneNinth, oneNinth, oneNinth },
        });
    }

    /// <summary>
    /// A weighted average blur that reduces image noise and detail, based on a Gaussian function
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> GaussianBlurKernel(int size) {
        const double oneSixteenth = 1.0/16.0;
        const double twoSixteenth = 2.0/16.0;
        const double fourSixteenth = 4.0/16.0;
        return new Matrix<double>(new double[,]{
            { oneSixteenth, twoSixteenth, oneSixteenth },
            { twoSixteenth, fourSixteenth, twoSixteenth },
            { oneSixteenth, twoSixteenth, oneSixteenth },
        });
    }

    /// <summary>
    /// Sobel kernel for edge detection, particularly for finding horizontal edges
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> SobelXKernel() {
        return new Matrix<double>(new double[,]{
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        });
    }

    /// <summary>
    /// Sobel kernel for edge detection, particularly for finding horizontal edges
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> SobelYKernel() {
        return new Matrix<double>(new double[,]{
            { -1, -2, -1 },
            { 0, 0, 0 },
            { 1, 2, 1 }
        });
    }

    /// <summary>
    /// Prewitt kernel for edge detection, similar to Sobel but with different weights
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> PrewittXKernel() {
        return new Matrix<double>(new double[,]{
            { -1, 0, 1 },
            { -1, 0, 1 },
            { -1, 0, 1 }
        });
    }

    /// <summary>
    /// Prewitt kernel for edge detection, similar to Sobel but with different weights
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> PrewittYKernel() {
        return new Matrix<double>(new double[,]{
            { -1, -1, -1 },
            { 0, 0, 0 },
            { 1, 1, 1 }
        });
    }

    /// <summary>
    /// Detects areas of rapid intensity change, commonly used for edge detection and sharpening
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> LaplacianKernel() {
        return new Matrix<double>(new double[,]{
            { 0, 1, 0 },
            { 1, -4, 1 },
            { 0, 1, 0 }
        });
    }

    /// <summary>
    /// Enhances the edges and fine details of an image by emphasizing differences between neighboring pixels
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> SharpeningKernel() {
        return new Matrix<double>(new double[,]{
            { 0, -1, 0 },
            {-1, 5, -1 },
            { 0, -1, 0 }
        });
    }

    /// <summary>
    /// Gives the image a 3D shadow effect, emphasizing edges and contours
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> EmbossKernel() {
        return new Matrix<double>(new double[,]{
            { -2, -1, 0 },
            {-1, 1, 1 },
            { 0, 1, 2 }
        });
    }
}