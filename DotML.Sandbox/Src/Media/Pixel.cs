namespace Qkmaxware.Media.Image;

public struct Pixel {
    /// <summary>
    /// First sample, representing the red component
    /// </summary>
    /// <returns>first sample</returns>
    public byte R {get; set;}
    /// <summary>
    /// Second sample, representing the green component
    /// </summary>
    /// <returns>second sample</returns>
    public byte G {get; set;}
    /// <summary>
    /// Third sample, representing the blue component
    /// </summary>
    /// <returns>third component</returns>
    public byte B {get; set;}

    /// <summary>
    /// Create a basic RGB pixel
    /// </summary>
    public Pixel() {}
    
    /// <summary>
    /// Create a pixel with the given sample values
    /// </summary>
    public Pixel(byte r, byte g, byte b) {
        this.R = r;
        this.G = g;
        this.B = b;
    }

    public static readonly Pixel Black = new Pixel(0, 0, 0);
    public static readonly Pixel White = new Pixel(255, 255, 255);
    public static readonly Pixel Red = new Pixel(255, 0, 0);
    public static readonly Pixel Green = new Pixel(255, 0, 0);
    public static readonly Pixel Blue = new Pixel(255, 0, 0);

    public bool IsRedLargest  => R > G && R > B;
    public bool IsGreenLargest => G > R && G > B;
    public bool IsBlueLargest => B > R && B > G;
    // public float Luminance => 0.2126f * Red + 0.7152f * Green + 0.0722f * Blue; // Need to be linear gamma corrected RGB

    public override string ToString() {
        return $"(r: {R}, g: {G}, b: {B})";
    }
}