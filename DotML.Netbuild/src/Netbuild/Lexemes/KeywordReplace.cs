using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordReplace : Keyword {
    public KeywordReplace() : base(@"\G\s*\b(?<value>REPLACE)\b\s*", "value") { }
}