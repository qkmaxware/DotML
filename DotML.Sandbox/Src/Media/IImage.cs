namespace Qkmaxware.Media.Image;

/// <summary>
/// Interface all images must implement
/// </summary>
public interface IImage { 
    // Specs
    public int Width {get;}
    public int Height {get;}

    // Data
    public Pixel[,]? Pixels {get;}
}

/// <summary>
/// An image entirely represented in memory
/// </summary>
public class MemoryImage : IImage {
    public Pixel[,] Pixels {get; private set;}

    public int Width => Pixels.GetLength(1);
    public int Height => Pixels.GetLength(0);

    public IEnumerable<Pixel> FlattenPixels() {
        for (var row = 0; row < Pixels.GetLength(0); row++) {
            for (var col = 0; col < Pixels.GetLength(1); col++) {
                yield return Pixels[row, col];
            }
        }
    }

    public MemoryImage (int width, int height) {
        this.Pixels = new Pixel[height,width];
    }

    public MemoryImage(Pixel[,] pixels) {
        this.Pixels = pixels;
    }

    public bool IsValidRow(int row) {
        return row >= 0 && row < this.Height;
    }

    public bool IsValidColumn(int column) {
        return column >= 0 && column < this.Width;
    }

    public bool IsValidCoordinate(int x, int y) {
        return IsValidColumn(x) && IsValidRow(y);
    }

    public void Stamp(int xOrig, int yOrig, MemoryImage stamp) {
        for (var y = 0; y < stamp.Height; y++) {
            for (var x = 0; x < stamp.Width; x++) {
                var cX = xOrig + x;
                var cY = yOrig + y;
                if (IsValidCoordinate(cX, cY)) {
                    this.Pixels[cX, cY] = stamp.Pixels[x, y];
                }
            }
        }
    }

    public MemoryImage ShiftBy(int x, int y) => ShiftBy(x, y, Pixel.Black);
    public MemoryImage ShiftBy(int x, int y, Pixel fill) {
        var next = new MemoryImage(this.Width, this.Height);

        for (var row = 0; row < this.Height; row++) {
            for (var col = 0; col < this.Width; col++) {
                var origX = col - x;
                var origY = row - y;

                if (IsValidCoordinate(origX, origY)) {
                    next.Pixels[row, col] = Pixels[origY, origX];
                } else {
                    next.Pixels[row, col] = fill;
                }
            }
        }

        return next;
    } 
    
    public MemoryImage Enlarge (int times) {
        times = Math.Max(1, times);
        MemoryImage next = new MemoryImage(this.Width * times, this.Height * times);
        for (var y = 0; y < this.Height; y++) {
            for (var x = 0; x < this.Width; x++) {
                var colour = this.Pixels[x, y];
                for (var xt = 0; xt < times; xt++) {
                    for (var yt = 0; yt < times; yt++) {
                        next.Pixels[x * times + xt, y * times + yt] = colour;
                    }
                }
            }
        }
        return next;
    }

    public MemoryImage[,] Slice(int hslices, int vslices) {
        hslices = Math.Max(1, hslices);
        vslices = Math.Max(1, vslices);

        var sliced = new MemoryImage[vslices,hslices];
        var width  = (int)Math.Round(Width  / (double)hslices);
        var height = (int)Math.Round(Height / (double)vslices);

        for (var y = 0; y < vslices; y++) {
            for (var x = 0; x < hslices; x++) {
                var img = new MemoryImage(width, height);

                for (var row = 0; row < height; row++) {
                    for (var col = 0; col < width; col++) {
                        if (IsValidCoordinate(x*width + col, y*height + row)) {
                            img.Pixels[row, col] = this.Pixels[y*height + row, x*width + col];
                        }
                    }
                }

                sliced[y, x] = img;
            }
        }

        return sliced;
    }
}