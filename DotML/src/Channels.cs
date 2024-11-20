using System.Numerics;

namespace DotML;

public static class ChannelsExtensions {
    public static Matrix<T> X<T>(this IEnumerable<Matrix<T>> channels) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        return channels.ElementAt(0);
    } 
    public static Matrix<T> Y<T>(this IEnumerable<Matrix<T>> channels) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        return channels.ElementAt(1);
    } 
    public static Matrix<T> Z<T>(this IEnumerable<Matrix<T>> channels) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        return channels.ElementAt(2);
    } 
    
    public static Matrix<T> Red<T>(this IEnumerable<Matrix<T>> channels) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        return channels.ElementAt(0);
    } 
    public static Matrix<T> Green<T>(this IEnumerable<Matrix<T>> channels) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        return channels.ElementAt(1);
    } 
    public static Matrix<T> Blue<T>(this IEnumerable<Matrix<T>> channels) where T:INumber<T>,IExponentialFunctions<T>,IRootFunctions<T> {
        return channels.ElementAt(2);
    } 
}