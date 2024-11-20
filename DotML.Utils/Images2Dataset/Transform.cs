using System.Drawing;

#pragma warning disable CA1416 // Only works in Windows

public abstract class Transform {
    public string Name {get; private set;}
    public Transform() {
        this.Name = GetType().Name;
    }
    public Transform Named(string name) {
        this.Name = name;
        return this;
    }
    public abstract IEnumerable<Bitmap> Apply(Bitmap original);
}

public class Identity : Transform {
    public override IEnumerable<Bitmap> Apply(Bitmap original) {
        Bitmap duplicate = new Bitmap(original);
        yield return duplicate;
    }
}

public class Scale : Transform {
    public float[] Scales;

    public Scale(float[] scales) {
        this.Scales = scales;
    }

    public Scale(float from, float to, float step = 1.0f) {
        var real_from = Math.Min(from, to);
        var real_end = Math.Max(from, to);
        var real_step = Math.Abs(step);

        int length = (int)Math.Ceiling((real_end - real_from) / real_step);
        var array = new float[length];
        for (int i = 0; i < length; i++) {
            var xi = real_from + i * real_step;
            if (xi > real_end)
                xi = real_end;
            array[i] = xi;
        }
        this.Scales = array;
    }

    public override IEnumerable<Bitmap> Apply(Bitmap original) {
        foreach (var scale in Scales) {  
            // Create a new Bitmap to hold the rotated image
            Bitmap scaledBitmap = new Bitmap(original.Width, original.Height);
            scaledBitmap.SetResolution(original.HorizontalResolution, original.VerticalResolution);

            int newWidth = (int)(original.Width * scale);
            int newHeight = (int)(original.Height * scale);

            // Create a Graphics object to perform the rotation
            using (Graphics graphics = Graphics.FromImage(scaledBitmap)) {
                graphics.Clear(Color.Black);

                // Set the rendering quality for better output
                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;

                 // Calculate the position to center the scaled image within the original dimensions
                int xOffset = (original.Width - newWidth) / 2;
                int yOffset = (original.Height - newHeight) / 2;

                // Scale the original image, keeping it centered within the original dimensions
                graphics.DrawImage(original, xOffset, yOffset, newWidth, newHeight);
            }

            yield return scaledBitmap;
        }
    }
}

public class Rotate : Transform {
    public float[] Angles;

    public Rotate(float[] Angles) {
        this.Angles = Angles;
    }

    public Rotate(float from, float to, float step = 1.0f) {
        var real_from = Math.Min(from, to);
        var real_end = Math.Max(from, to);
        var real_step = Math.Abs(step);

        int length = (int)Math.Ceiling((real_end - real_from) / real_step);
        var array = new float[length];
        for (int i = 0; i < length; i++) {
            var xi = real_from + i * real_step;
            if (xi > real_end)
                xi = real_end;
            array[i] = xi;
        }
        this.Angles = array;
    }


    public override IEnumerable<Bitmap> Apply(Bitmap original) {
        foreach (var angle in Angles) {
            // Create a new Bitmap to hold the rotated image
            Bitmap rotatedBitmap = new Bitmap(original.Width, original.Height);
            rotatedBitmap.SetResolution(original.HorizontalResolution, original.VerticalResolution);

            // Create a Graphics object to perform the rotation
            using (Graphics graphics = Graphics.FromImage(rotatedBitmap)) {
                graphics.Clear(Color.Black);

                // Set the rendering quality for better output
                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;

                // Set the rotation point to the center of the original image
                graphics.TranslateTransform(original.Width / 2, original.Height / 2);

                // Rotate the image by the specified angle
                graphics.RotateTransform(angle);

                // Move the image back to its original position after rotation
                graphics.TranslateTransform(-original.Width / 2, -original.Height / 2);

                // Draw the original image onto the rotated Bitmap
                graphics.DrawImage(original, 0, 0);
            }

            yield return rotatedBitmap;
        }
    }
}

public class Flip : Transform {

    public bool X {get; init;}
    public bool Y {get; init;}
    public bool XY {get; init;}

    public Flip(bool xflip, bool yflip, bool xyflip) {
        this.X = xflip;
        this.Y = yflip;
        this.XY = xyflip;
    }

    public override IEnumerable<Bitmap> Apply(Bitmap original) {
        // Create a new Bitmap to hold the flipped image(s)
        if (X) {
            Bitmap scaledBitmap = new Bitmap(original);
            scaledBitmap.RotateFlip(RotateFlipType.RotateNoneFlipX);
            yield return scaledBitmap;
        }
        if (Y) {
            Bitmap scaledBitmap = new Bitmap(original);
            scaledBitmap.RotateFlip(RotateFlipType.RotateNoneFlipY);
            yield return scaledBitmap;
        }
        if (XY) {
            Bitmap scaledBitmap = new Bitmap(original);
            scaledBitmap.RotateFlip(RotateFlipType.RotateNoneFlipXY);
            yield return scaledBitmap;
        }
    }
}