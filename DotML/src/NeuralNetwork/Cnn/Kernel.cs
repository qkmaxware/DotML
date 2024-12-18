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
    /// Sobel kernel for edge detection, particularly for finding horizontal edges
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> SobelKernel() {
        return new Matrix<double>(new double[,]{
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        });
    }

    /// <summary>
    /// Prewitt kernel for edge detection, similar to Sobel but with different weights
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> PrewittKernel() {
        return new Matrix<double>(new double[,]{
            { -1, 0, 1 },
            { -1, 0, 1 },
            { -1, 0, 1 }
        });
    }

    /// <summary>
    /// This kernel leaves the image unchanged, effectively performing no operation
    /// </summary>
    /// <returns>matrix</returns>
    public static Matrix<double> IdentityKernel() {
        return new Matrix<double>(new double[,]{
            { 0, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 0 }
        });
    }
}