using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordInput : Keyword {
    public KeywordInput() : base(@"\G\s*\b(?<value>INPUT)\b\s*", "value") { }
}