# DotML NetFlow
A command line utility for:
1. Building Neural Networks
2. Managing Neural Networks
3. Training Neural Networks
4. Locally run Neural Networks

## Install
1. Clone or Download repo
2. Using dotnet 8 or newer run the following commands
    - dotnet pack DotML.Cli.csproj -c Release
    - dotnet tool install --configfile none.nuget.config --global --add-source ./nupkg netflow
  
## Uninstall
1. Using dotnet cli run the following commands
    - dotnet tool uninstall --global netflow

## Update 
1. Just combine Install and uninstall such that
    - dotnet tool uninstall --global netflow; dotnet pack DotML.Cli.csproj -c Release; dotnet tool install --configfile none.nuget.config --global --add-source ./nupkg netflow