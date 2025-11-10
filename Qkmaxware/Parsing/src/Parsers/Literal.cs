using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Qkmaxware.Parsing {

    /// <summary>
    /// Static parsers for literals
    /// </summary>
    public static class Literal
    {

        /// <summary>
        /// Parse a boolean value
        /// </summary>
        public static Parser<bool> Boolean()
        {
            return Text.Is("true").Or(Text.Is("false")).Map(value => System.Boolean.Parse(value));
        }

        /// <summary>
        /// Parse a whole number (positive integers including 0)
        /// </summary>
        public static Parser<long> Whole()
        {
            return Character.Digit().OneOrMore().Unless(digits => digits[0] == '0' && digits.Count > 1).Map(digits => long.Parse(string.Concat(digits)));
        }

        /// <summary>
        /// Parse a fractional portion of a decimal number
        /// </summary>
        private static Parser<double> Fraction()
        {
            return Character.Is('.').Then(Whole().Map(digits => Double.Parse("0." + digits)));
        }

        /// <summary>
        /// Parse a real number
        /// </summary>
        public static Parser<double> Real()
        {
            return Whole().Then(
                whole => Fraction().Map(frac => whole + frac)
                );
        }

        private static double power(char sign, double @base, long exp)
        {
            return (
                sign == '-'
                ? @base * Math.Pow(10, -(double)exp)
                : @base * Math.Pow(10, (double)exp)
            );
        }

        /// <summary>
        /// Parse a number in scientific format
        /// </summary>
        public static Parser<double> Scientific()
        {
            return Real().Then(
                value => (Character.Is('E').Or(Character.Is('e'))).Then(
                    e => (Character.Is('-').Or(Character.Is('+')).Optional()).Then(
                        sign => Whole().Map(exp => power(sign.Match((some) => some.Value, (none) => '+'), value, exp))
                    )
                )
            );
        }

        /// <summary>
        /// Parse a number in any of the standard formats
        /// </summary>
        public static Parser<double> Number()
        {
            return Scientific().Or(Real()).Or(Whole().Map(integer => (double)integer));
        }

        /// <summary>
        /// Parse a TimeSpan in the format of hh:mm:ss[.fffffff]
        /// </summary>
        public static Parser<TimeSpan> Timespan()
        {
            return Character.Digit().Repeat(2).Then(
                hours => Character.Digit().Repeat(2).Then(
                    minutes => Character.Digit().Repeat(2).Then(
                        seconds => Fraction().Optional().Map(
                            fractionalSeconds =>
                            {
                                var hh = int.Parse(string.Concat(hours));
                                var mm = int.Parse(string.Concat(minutes));
                                var ss = int.Parse(string.Concat(seconds));
                                var fffffff = fractionalSeconds;

                                return TimeSpan.FromHours(hh) + TimeSpan.FromMinutes(mm) + TimeSpan.FromSeconds(ss + fffffff.Match((some) => some.Value, (none) => 0));
                            }
                        )
                    )
                )
            );
        }

        /// <summary>
        /// Parse a standard programming language based identifier [a-zA-Z\200-\377_][a-zA-Z\200-\377_0-9]*
        /// </summary>
        /// <returns>string</returns>
        public static Parser<string> Identifier()
        {
            var valid_first = Character.Range('a', 'z').Or(Character.Range('A', 'Z')).Or(Character.Range('\u0200', '\u0377')).Or(Character.Is('_'));
            var valid_second = valid_first.Or(Character.Digit());

            return valid_first.ThenMap(valid_second.ZeroOrMore(), (lhs, rhs) =>
            {
                rhs.Insert(0, lhs); // Prepend the first char
                return new string(CollectionsMarshal.AsSpan(rhs));
            });
        }

}

}