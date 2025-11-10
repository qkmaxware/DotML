using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordArg : Keyword {
    public KeywordArg() : base(@"\G\s*\b(?<value>ARG)\b\s*", "value") { }
}