using DotML;
using DotML.Sandbox.Components.Layout;
using Qkmaxware.Media.Image;

namespace DotML.Sandbox.Data;


/// <summary>
/// Convert Greyscale art to a vector by converting from 8bit RGB to -1 to 1 greyscale
/// </summary>
public class GreyscaleImgVectorizor : IFeatureExtractor<Pixel[]> {

    private byte greyscale(Pixel pixel) {
        return Math.Max(pixel.R, Math.Max(pixel.G, pixel.B)); // (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
    }

    private double rescale(byte b) {
        var sample = b / 255.0;
        return  (1.0 * sample) + (-1.0 * (1.0 - sample));
    }

    public Vec<double> ToVector(Pixel[] value) {
        var vecdata = new double[value.Length];
        for (var i = 0; i < vecdata.Length; i++) {
            vecdata[i] = rescale(greyscale(value[i]));
        }
        return vecdata;
    }
}
