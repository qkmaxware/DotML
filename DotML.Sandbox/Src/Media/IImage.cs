using System.Drawing;

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
    public Pixel GetPixelOrDefault(int x, int y, Pixel def);
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
        this.Pixels = new Pixel[height, width];
    }

    public MemoryImage(Pixel[,] pixels) {
        this.Pixels = pixels;
    }

    public Pixel GetPixelOrDefault(int x, int y, Pixel def) {
        if (x < 0 || x >= Width || y < 0 || y >= Height)
            return def;
        
        return this.Pixels[y, x];
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
                    this.Pixels[cY, cX] = stamp.Pixels[y, x];
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

    public IEnumerable<MemoryImage> SliceAutoWidth(Pixel background) {
        // Find first vscan that only contains background pixels
        var startx = 0;
        for (var x = 0; x < this.Width; x++) {
            bool is_vscan_all_background = true;
            for (var y = 0; y < this.Height; y++) {
                var colour = this.Pixels[y, x];
                if (colour != background) {
                    is_vscan_all_background = false;
                    break;
                }
            }

            if (!is_vscan_all_background)
                continue;

            if (startx == x) {
                // Image has a width of 0, skip it
                startx = x + 1;
                continue;
            }
            else {
                // Do copy 
                MemoryImage img = new MemoryImage(x - startx, this.Height);
                for (var x2 = startx; x2 < x; x2++) {
                    for (var y2 = 0; y2 < this.Height; y2++) {
                        img.Pixels[y2, x2 - startx] = this.Pixels[y2, x2];
                    }
                }
                yield return img;
                startx = x + 1;
                continue;
            }
        }

        if (startx != this.Width) {
            // Do copy 
            MemoryImage img = new MemoryImage(this.Width - startx, this.Height);
            for (var x2 = startx; x2 < this.Width; x2++) {
                for (var y2 = 0; y2 < this.Height; y2++) {
                    img.Pixels[y2, x2 - startx] = this.Pixels[y2, x2];
                }
            }
            yield return img;
        }
    }

    public MemoryImage CropToTarget(Pixel background) {
        // Determine the "background colour"
        // Find the region containing pixels that aren't the background colour
        var xmin = 0; var ymin = 0;
        var xmax = this.Width - 1; var ymax = this.Height - 1;
        
        for (var y = 0; y < this.Height; y++) {
            for (var x = 0; x < this.Width; x++) {
                var colour = this.Pixels[y, x];
                if (colour != background) {
                    ymin = y;
                    goto max_height;
                }
            }
        }
        max_height:
        for (var y = this.Height - 1; y >= ymin; y--) {
            for (var x = 0; x < this.Width; x++) {
                var colour = this.Pixels[y, x];
                if (colour != background) {
                    ymax = y;
                    goto done_y;
                }
            }
        }
        done_y:

        for (var x = 0; x < this.Width; x++) {
            for (var y = 0; y < this.Height; y++) {
                var colour = this.Pixels[y, x];
                if (colour != background) {
                    xmin = x;
                    goto max_width;
                }
            }
        }
        max_width:
        for (var x = this.Width - 1; x >= xmin; x--) {
            for (var y = 0; y < this.Height; y++) {
                var colour = this.Pixels[y, x];
                if (colour != background) {
                    xmax = x;
                    goto done_x;
                }
            }
        }
        done_x:

        // Create a new image with just those pixels in the region
        var result = new MemoryImage(xmax - xmin + 1, ymax - ymin + 1);
        for (var y = ymin; y <= ymax; y++) {
            for (var x = xmin; x <= xmax; x++) {
                result.Pixels[y - ymin, x - xmin] = this.Pixels[y, x];
            }
        }

        return result;
    }
    
    public MemoryImage Enlarge (int times) {
        times = Math.Max(1, times);
        MemoryImage next = new MemoryImage(this.Width * times, this.Height * times);
        for (var y = 0; y < this.Height; y++) {
            for (var x = 0; x < this.Width; x++) {
                var colour = this.Pixels[y, x];
                for (var xt = 0; xt < times; xt++) {
                    for (var yt = 0; yt < times; yt++) {
                        next.Pixels[y * times + yt, x * times + xt] = colour;
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