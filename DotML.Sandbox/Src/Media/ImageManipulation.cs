namespace Qkmaxware.Media.Image;

public interface IImageScaler {
    public IImage ScaleContents(IImage image, double factor);
}

public class NearestNeighbourScaler : IImageScaler {
    public IImage ScaleContents(IImage image, double factor) {
        int height = image.Height;
        int width = image.Width;
        var scaled = new Pixel[height,width];

        if (image.Pixels is null)
            return new MemoryImage(scaled);

        double centerX = width / 2.0;
        double centerY = height / 2.0;

        for (var row = 0; row < height; row++) {
            for (var col = 0; col < width; col++) {
                // Original position
                double scaledX = (col - centerX) * factor + centerX;
                double scaledY = (row - centerY) * factor + centerY;

                // Round down
                int originalX = (int)Math.Round(scaledX);
                int originalY = (int)Math.Round(scaledY);

                if (originalX >= 0 && originalX < width && originalY >= 0 && originalY < height) {
                    scaled[row, col] = image.Pixels[originalY, originalX];
                } else {
                    scaled[row, col] = Pixel.Black;
                }
            }
        }

        return new MemoryImage(scaled);
    }
}

public class BicubicScaler : IImageScaler {

    private double BicubicWeight(double x, double y) {
        double a = -0.5; // Bicubic interpolation parameter
        return ((a + 2) * Math.Pow(Math.Abs(x), 3) - (a + 3) * Math.Pow(Math.Abs(x), 2) + 1) *
               ((a + 2) * Math.Pow(Math.Abs(y), 3) - (a + 3) * Math.Pow(Math.Abs(y), 2) + 1);
    }

    private Pixel InterpolateColor(IImage image, double[,] weights, int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3) {
        double totalWeight = 0.0;
        int r = 0;
        int g = 0;
        int b = 0;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int sampleX = Math.Clamp(x1 + j, 0, image.Width - 1);
                int sampleY = Math.Clamp(y1 + i, 0, image.Height - 1);

                double weight = weights[i, j];
                totalWeight += weight;

                if (weight > 0)
                {
                    Pixel pixelColor = image.Pixels[sampleY, sampleX];
                    r = (byte)(r + pixelColor.R * weight);
                    g = (byte)(g + pixelColor.G * weight);
                    b = (byte)(b + pixelColor.B * weight);
                }
            }
        }

        // Normalize the color
        if (totalWeight > 0) {
            return new Pixel {
                R = (byte)Math.Clamp((int)(r / totalWeight), 0, 255),
                G = (byte)Math.Clamp((int)(g / totalWeight), 0, 255),
                B = (byte)Math.Clamp((int)(b / totalWeight), 0, 255)
            };
        }

        return Pixel.Black; // Return a default color if no weight is found
    }

    public IImage ScaleContents(IImage image, double factor) {
        int height = image.Height;
        int width = image.Width;
        var scaled = new Pixel[height,width];

        if (image.Pixels is null)
            return new MemoryImage(scaled);

        double centerX = width / 2.0;
        double centerY = height / 2.0;
        double[,] weights = new double[4, 4];

        for (var row = 0; row < height; row++) {
            for (var col = 0; col < width; col++) {
                // Original position
                double scaledX = (col - centerX) * factor + centerX;
                double scaledY = (row - centerY) * factor + centerY;

                // Get surrounding pixels
                int x0 = (int)Math.Floor(scaledX) - 1;
                int x1 = (int)Math.Floor(scaledX);
                int x2 = (int)Math.Ceiling(scaledX);
                int x3 = x2 + 1;

                int y0 = (int)Math.Floor(scaledY) - 1;
                int y1 = (int)Math.Floor(scaledY);
                int y2 = (int)Math.Ceiling(scaledY);
                int y3 = y2 + 1;

                //Bicubic weights
                for (int i = -1; i <= 2; i++) {
                    for (int j = -1; j <= 2; j++) {
                        weights[i + 1, j + 1] = BicubicWeight(scaledX - (x1 + j), scaledY - (y1 + i));
                    }
                }

                // Interpolate colour
                var interpolated = InterpolateColor(image, weights, x0, y0, x1, y1, x2, y2, x3, y3);

                // Assign
                scaled[row, col] = interpolated;
            }
        }

        return new MemoryImage(scaled);
    }
}

public interface IImageRotator {
    public IImage RotateContents(IImage image, double radians);
}

public class BicubicRotator : IImageRotator {
    private double BicubicWeight(double x, double y) {
        double a = -0.5; // Bicubic interpolation parameter
        return ((a + 2) * Math.Pow(Math.Abs(x), 3) - (a + 3) * Math.Pow(Math.Abs(x), 2) + 1) *
               ((a + 2) * Math.Pow(Math.Abs(y), 3) - (a + 3) * Math.Pow(Math.Abs(y), 2) + 1);
    }

    private Pixel InterpolateColor(IImage image, double[,] weights, int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3) {
        double totalWeight = 0.0;
        int r = 0;
        int g = 0;
        int b = 0;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int sampleX = Math.Clamp(x1 + j, 0, image.Width - 1);
                int sampleY = Math.Clamp(y1 + i, 0, image.Height - 1);

                double weight = weights[i, j];
                totalWeight += weight;

                if (weight > 0)
                {
                    Pixel pixelColor = image.Pixels[sampleY, sampleX];
                    r = (byte)(r + pixelColor.R * weight);
                    g = (byte)(g + pixelColor.G * weight);
                    b = (byte)(b + pixelColor.B * weight);
                }
            }
        }

        // Normalize the color
        if (totalWeight > 0) {
            return new Pixel {
                R = (byte)Math.Clamp((int)(r / totalWeight), 0, 255),
                G = (byte)Math.Clamp((int)(g / totalWeight), 0, 255),
                B = (byte)Math.Clamp((int)(b / totalWeight), 0, 255)
            };
        }

        return Pixel.Black; // Return a default color if no weight is found
    }

    public IImage RotateContents(IImage image, double radians) {
        int height = image.Height;
        int width = image.Width;
        var rotated = new Pixel[height,width];

        if (image.Pixels is null)
            return new MemoryImage(rotated);

        double centerX = width / 2.0;
        double centerY = height / 2.0;
        double[,] weights = new double[4, 4];

        for (var row = 0; row < height; row++) {
            for (var col = 0; col < width; col++) {
                // Original position
                double x = (col - centerX);
                double y = (row - centerY);
                
                double rotatedX = Math.Cos(radians) * x - Math.Sin(radians) * y + centerX;
                double rotatedY = Math.Sin(radians) * x + Math.Cos(radians) * y + centerY;

                // Get surrounding pixels
                int x0 = (int)Math.Floor(rotatedX) - 1;
                int x1 = (int)Math.Floor(rotatedX);
                int x2 = (int)Math.Ceiling(rotatedX);
                int x3 = x2 + 1;

                int y0 = (int)Math.Floor(rotatedY) - 1;
                int y1 = (int)Math.Floor(rotatedY);
                int y2 = (int)Math.Ceiling(rotatedY);
                int y3 = y2 + 1;

                //Bicubic weights
                for (int i = -1; i <= 2; i++) {
                    for (int j = -1; j <= 2; j++) {
                        weights[i + 1, j + 1] = BicubicWeight(rotatedX - (x1 + j), rotatedY - (y1 + i));
                    }
                }

                // Interpolate colour
                var interpolated = InterpolateColor(image, weights, x0, y0, x1, y1, x2, y2, x3, y3);

                // Assign
                rotated[row, col] = interpolated;
            }
        }

        return new MemoryImage(rotated);
    }
}

public interface IImageShearer {
    public IImage ShearContents(IImage image, double shearX, double shearY);
}

public class BicubicShearer : IImageShearer {
    private double BicubicWeight(double x, double y) {
        double a = -0.5; // Bicubic interpolation parameter
        return ((a + 2) * Math.Pow(Math.Abs(x), 3) - (a + 3) * Math.Pow(Math.Abs(x), 2) + 1) *
               ((a + 2) * Math.Pow(Math.Abs(y), 3) - (a + 3) * Math.Pow(Math.Abs(y), 2) + 1);
    }

    private Pixel InterpolateColor(IImage image, double[,] weights, int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3) {
        double totalWeight = 0.0;
        int r = 0;
        int g = 0;
        int b = 0;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int sampleX = Math.Clamp(x1 + j, 0, image.Width - 1);
                int sampleY = Math.Clamp(y1 + i, 0, image.Height - 1);

                double weight = weights[i, j];
                totalWeight += weight;

                if (weight > 0)
                {
                    Pixel pixelColor = image.Pixels[sampleY, sampleX];
                    r = (byte)(r + pixelColor.R * weight);
                    g = (byte)(g + pixelColor.G * weight);
                    b = (byte)(b + pixelColor.B * weight);
                }
            }
        }

        // Normalize the color
        if (totalWeight > 0) {
            return new Pixel {
                R = (byte)Math.Clamp((int)(r / totalWeight), 0, 255),
                G = (byte)Math.Clamp((int)(g / totalWeight), 0, 255),
                B = (byte)Math.Clamp((int)(b / totalWeight), 0, 255)
            };
        }

        return Pixel.Black; // Return a default color if no weight is found
    }

    public IImage ShearContents(IImage image, double shearX, double shearY) {
        int height = image.Height;
        int width = image.Width;
        var sheared = new Pixel[height,width];

        if (image.Pixels is null)
            return new MemoryImage(sheared);

        double centerX = width / 2.0;
        double centerY = height / 2.0;
        double[,] weights = new double[4, 4];

        for (var row = 0; row < height; row++) {
            for (var col = 0; col < width; col++) {
                // Original position
                double x = (col - centerX);
                double y = (row - centerY);
                
                // Apply shearing
                double shearedX = x + shearX * y;
                double shearedY = shearY * x + y;
                shearedX += centerX;
                shearedY += centerY;

                // Get surrounding pixels
                int x0 = (int)Math.Floor(shearedX) - 1;
                int x1 = (int)Math.Floor(shearedX);
                int x2 = (int)Math.Ceiling(shearedX);
                int x3 = x2 + 1;

                int y0 = (int)Math.Floor(shearedY) - 1;
                int y1 = (int)Math.Floor(shearedY);
                int y2 = (int)Math.Ceiling(shearedY);
                int y3 = y2 + 1;

                //Bicubic weights
                for (int i = -1; i <= 2; i++) {
                    for (int j = -1; j <= 2; j++) {
                        weights[i + 1, j + 1] = BicubicWeight(shearedX - (x1 + j), shearedY - (y1 + i));
                    }
                }

                // Interpolate colour
                var interpolated = InterpolateColor(image, weights, x0, y0, x1, y1, x2, y2, x3, y3);

                // Assign
                sheared[row, col] = interpolated;
            }
        }

        return new MemoryImage(sheared);
    }
}

public interface IImageNoisier {
    public IImage AddNoise(IImage image, double mean, double stdDev);
}

public class GaussianNoisier : IImageNoisier {
    private Random rng = new Random();

    private double GenerateGaussianNoise(double mean, double stdDev) {
        // Box-Muller transform to generate Gaussian noise
        double u1 = 1.0 - rng.NextDouble(); // Uniform(0,1] random doubles
        double u2 = 1.0 - rng.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2); // Gaussian(0,1)
        return z0 * stdDev + mean; // Transform to desired mean and standard deviation
    }

    public IImage AddNoise(IImage image, double mean, double stdDev) {
        int height = image.Height;
        int width = image.Width;
        var next = new Pixel[height,width];

        if (image.Pixels is null)
            return new MemoryImage(next);

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                var originalColor = image.Pixels[row, col];

                // Apply Gaussian noise to each channel
                int r = Math.Clamp((int)(originalColor.R + GenerateGaussianNoise(mean, stdDev)), 0, 255);
                int g = Math.Clamp((int)(originalColor.G + GenerateGaussianNoise(mean, stdDev)), 0, 255);
                int b = Math.Clamp((int)(originalColor.B + GenerateGaussianNoise(mean, stdDev)), 0, 255);

                // Assign pixel
                next[row, col] = new Pixel {
                    R = (byte)r,
                    G = (byte)g,
                    B = (byte)b
                };
            }
        }

        return new MemoryImage(next);
    }
}

public class BicubicImageTransformer {
    private BicubicScaler scaler = new BicubicScaler();
    private BicubicRotator rotator = new BicubicRotator();
    private BicubicShearer shearer = new BicubicShearer();

    public IImage ScaleRotateShear(IImage image, double scale, double rotation, double shearX, double shearY) => shearer.ShearContents(rotator.RotateContents(scaler.ScaleContents(image, scale), rotation), shearX, shearY);
}