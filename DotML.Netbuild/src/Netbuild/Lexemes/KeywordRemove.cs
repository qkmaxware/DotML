using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordRemove : Keyword {
    public KeywordRemove() : base(@"\G\s*\b(?<value>REMOVE)\b\s*", "value") { }
}