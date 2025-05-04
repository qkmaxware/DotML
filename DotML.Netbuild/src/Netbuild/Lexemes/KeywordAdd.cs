using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordAdd : Keyword {
    public KeywordAdd() : base(@"\G\s*\b(?<value>ADD)\b\s*", "value") { }
}