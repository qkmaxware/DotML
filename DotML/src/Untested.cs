namespace DotML;

/// <summary>
/// Attribute to indicate that the functionality of a component has not been tested yet
/// </summary>
[Obsolete("Usage of an untested class", error: false)] // This makes it so we get compiler warnings whenever this is used
public class UntestedAttribute : System.Attribute {}

/// <summary>
/// Attribute to indicate that the a component is still in active development
/// </summary>
[Obsolete("Usage of a class which is still a work in progress and should not be used", error: false)] // This makes it so we get compiler warnings whenever this is used
public class WorkInProgress : System.Attribute {}