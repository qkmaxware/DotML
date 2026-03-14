using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordName : Keyword {
    public KeywordName() : base(@"\G\s*\b(?<value>NAME|LABEL)\b\s*", "value") { }
}