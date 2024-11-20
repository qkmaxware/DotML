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
    public static readonly Pixel Pink = new Pixel(255, 192, 203);
    public static readonly Pixel Orange = new Pixel(255, 165, 0);
    public static readonly Pixel Yellow = new Pixel(255, 255, 0);
    public static readonly Pixel Violet = new Pixel(238, 130, 238);
    public static readonly Pixel Purple = new Pixel(128, 0, 128);
    public static readonly Pixel Teal = new Pixel(0, 128, 128);
    public static readonly Pixel Cyan = new Pixel(0, 255, 255);
    public static readonly Pixel Turquoise = new Pixel(64, 224, 208);

    public bool IsRedLargest  => R > G && R > B;
    public bool IsGreenLargest => G > R && G > B;
    public bool IsBlueLargest => B > R && B > G;

    public float Luminance => 0.2126f * R + 0.7152f * G + 0.0722f * B; 

    public override string ToString() {
        return $"(r: {R}, g: {G}, b: {B})";
    }
}