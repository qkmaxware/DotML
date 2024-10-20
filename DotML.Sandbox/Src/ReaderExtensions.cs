using System.IO;
using System.Text;

namespace Qkmaxware.Media;

public static class TextReaderExtensions {
    public static int ReadInt(this TextReader reader) {
        StringBuilder sb = new StringBuilder();
        if (reader.Peek() == -1)
            return 0;
        if (reader.Peek() == '-') {
            sb.Append('-');
            reader.Read();
        }
        while (reader.Peek() != -1 && char.IsDigit((char)reader.Peek())) {
            sb.Append((char)reader.Read());
        }
        return int.Parse(sb.ToString());
    }
    public static uint ReadUInt(this TextReader reader) {
        StringBuilder sb = new StringBuilder();
        if (reader.Peek() == -1)
            return 0;
        while (reader.Peek() != -1 && char.IsDigit((char)reader.Peek())) {
            sb.Append((char)reader.Read());
        }
        return uint.Parse(sb.ToString());
    }
}

public static class BinaryReaderExtensions {
    public static int ReadAsciiInt(this BinaryReader reader) {
        StringBuilder sb = new StringBuilder();
        if (reader.PeekChar() == -1)
            return 0;
        if (reader.PeekChar() == '-') {
            sb.Append('-');
            reader.Read();
        }
        while (reader.PeekChar() != -1 && char.IsDigit((char)reader.PeekChar())) {
            sb.Append((char)reader.Read());
        }
        return int.Parse(sb.ToString());
    }
    public static uint ReadAsciiUInt(this BinaryReader reader) {
        StringBuilder sb = new StringBuilder();
        if (reader.PeekChar() == -1)
            return 0;
        while (reader.PeekChar() != -1 && char.IsDigit((char)reader.PeekChar())) {
            sb.Append((char)reader.Read());
        }
        return uint.Parse(sb.ToString());
    }
}