using System.Text.RegularExpressions;

namespace DotML.Network.IO.Netbuild;

internal class KeywordPretrain : Keyword {
    public KeywordPretrain() : base(@"\G\s*\b(?<value>PRETRAIN)\b\s*", "value") { }
}