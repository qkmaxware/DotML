namespace DotML.Playground.Common.Data;

public class BitmapFormat {
    public void SaveTo(BinaryWriter writer, IImage image) {
        if (image.Pixels is null) {
            throw new ArgumentNullException(nameof(image.Pixels), "Image pixels cannot be null.");
        }
        
        int width = image.Width;
        int height = image.Height;

        int rowPadding = (4 - (width * 3) % 4) % 4;

        int fileSize = 54 + (width * 3 + rowPadding) * height;
        int dataOffset = 54;

        // === BMP Header (14 bytes) ===
        writer.Write((byte)'B');
        writer.Write((byte)'M');
        writer.Write(fileSize);
        writer.Write(0); // Reserved
        writer.Write(dataOffset); // Pixel data offset

        // === DIB Header (40 bytes - BITMAPINFOHEADER) ===
        writer.Write(40);               // Header size
        writer.Write(width);
        writer.Write(height);
        writer.Write((short)1);         // Planes
        writer.Write((short)24);        // Bits per pixel
        writer.Write(0);                // Compression (none)
        writer.Write(0);                // Image size (can be 0 for no compression)
        writer.Write(0);                // X pixels per meter
        writer.Write(0);                // Y pixels per meter
        writer.Write(0);                // Total colors
        writer.Write(0);                // Important colors

        // === Pixel Data (bottom-up) ===
        for (int y = height - 1; y >= 0; y--) // BMP stores pixels bottom-to-top
        {
            for (int x = 0; x < width; x++) {
                var color = image.Pixels[y, x];
                writer.Write(color.B); // BMP uses BGR format
                writer.Write(color.G);
                writer.Write(color.R);
            }

            // Write padding
            for (int p = 0; p < rowPadding; p++)
                writer.Write((byte)0);
        }
    }
}