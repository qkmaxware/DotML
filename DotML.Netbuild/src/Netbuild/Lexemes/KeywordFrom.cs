using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordFrom : Keyword {
    public KeywordFrom() : base(@"\G\s*\b(?<value>FROM)\b\s*", "value") { }
}