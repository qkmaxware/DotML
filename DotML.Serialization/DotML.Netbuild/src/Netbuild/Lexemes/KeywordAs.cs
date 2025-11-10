using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordAs : Keyword {
    public KeywordAs() : base(@"\G\s*\b(?<value>AS)\b\s*", "value") { }
}