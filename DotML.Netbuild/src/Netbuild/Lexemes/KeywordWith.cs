using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordWith : Keyword {
    public KeywordWith() : base(@"\G\s*\b(?<value>WITH)\b\s*", "value") { }
}