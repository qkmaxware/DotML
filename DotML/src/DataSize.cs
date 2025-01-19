namespace DotML;

/// <summary>
/// Data units of measure
/// </summary>
public enum DataUnit : byte {
    Bytes = 0,
    Decabyte = 1,
    Hectobyte = 2,
    Kilobyte = 3,
    Megabyte = 4,
    Gigabyte = 5,
}

public static class DataUnitExtensions {
    public static int Factor(this DataUnit uom) {
        return uom switch {
            DataUnit.Decabyte => 10,
            DataUnit.Hectobyte => 100,
            DataUnit.Kilobyte => 1_000,
            DataUnit.Megabyte => 1_000_000,
            DataUnit.Gigabyte => 1_000_000_000,
            _ => 1
        };
    }
    public static string Prefix(this DataUnit uom) {
        return uom switch {
            DataUnit.Decabyte => "deca",
            DataUnit.Hectobyte => "hecto",
            DataUnit.Kilobyte => "kilo",
            DataUnit.Megabyte => "mega",
            DataUnit.Gigabyte => "giga",
            _ => string.Empty
        };
    }
    public static string Symbol(this DataUnit uom) {
        return uom switch {
            DataUnit.Decabyte => "dab",
            DataUnit.Hectobyte => "hb",
            DataUnit.Kilobyte => "Kb",
            DataUnit.Megabyte => "Mb",
            DataUnit.Gigabyte => "Gb",
            _ => "b"
        };
    }
}

/// <summary>
/// Struct representing the size of some data
/// </summary>
public struct DataSize {
    private double value;
    private DataUnit uom;

    public DataSize(double value, DataUnit uom) {
        this.value = value;
        this.uom = uom;
    }

    public static DataSize FromValues32(int value_count) {
        // This assumes that each value is 4bytes = 32bits
        long totalBytes = value_count * 4L;

        // Choose the appropriate unit based on the total bytes.
        if (totalBytes >= DataUnit.Gigabyte.Factor()) {
            // If the size is at least 1GB, use Gigabytes.
            return new DataSize(totalBytes / DataUnit.Gigabyte.Factor(), DataUnit.Gigabyte);
        } else if (totalBytes >= DataUnit.Megabyte.Factor()) {
            // If the size is at least 1MB, use Megabytes.
            return new DataSize(totalBytes / DataUnit.Megabyte.Factor(), DataUnit.Megabyte);
        } else if (totalBytes >= DataUnit.Kilobyte.Factor()) {
            // If the size is at least 1KB, use Kilobytes.
            return new DataSize(totalBytes / DataUnit.Kilobyte.Factor(), DataUnit.Kilobyte);
        } else {
            // If it's less than 1KB, use Bytes.
            return new DataSize(totalBytes, DataUnit.Bytes);
        }
    }

    public static DataSize FromValues64(int value_count) {
        // This assumes that each value is 8bytes = 64bits
        long totalBytes = value_count * 8L;

        // Choose the appropriate unit based on the total bytes.
        if (totalBytes >= DataUnit.Gigabyte.Factor()) {
            // If the size is at least 1GB, use Gigabytes.
            return new DataSize(totalBytes / DataUnit.Gigabyte.Factor(), DataUnit.Gigabyte);
        } else if (totalBytes >= DataUnit.Megabyte.Factor()) {
            // If the size is at least 1MB, use Megabytes.
            return new DataSize(totalBytes / DataUnit.Megabyte.Factor(), DataUnit.Megabyte);
        } else if (totalBytes >= DataUnit.Kilobyte.Factor()) {
            // If the size is at least 1KB, use Kilobytes.
            return new DataSize(totalBytes / DataUnit.Kilobyte.Factor(), DataUnit.Kilobyte);
        } else {
            // If it's less than 1KB, use Bytes.
            return new DataSize(totalBytes, DataUnit.Bytes);
        }
    }

    /// <summary>
    /// Get the value of the size in the given units of measure
    /// </summary>
    /// <param name="uom">units of measure</param>
    /// <returns>value in the given units of measure</returns>
    public double ValueAs(DataUnit uom) {
        // Find the current power of 10s
        var current_power = (int)this.uom;
        var target_power = (int)uom;   

        // Calculate the difference
        var difference = current_power - target_power;

        // Convert
        return this.value * Math.Pow(10, difference);
    }

    public override string ToString() {
        return this.value.ToString("0.##") + uom.Symbol();
    }
}