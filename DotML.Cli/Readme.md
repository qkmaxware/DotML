# DotML NetFlow
A command line utility for:
1. Building Neural Networks
```ps
> netflow build my_network.netbuild --tag alexnet
```
2. Managing Neural Networks
```ps
> netflow list

> netflow rm my_network

> netflow tag --add "custom-tag"
```
3. Training Neural Networks
```ps
> netflow fit my_network --data-training "training-vectors.bin"
```
4. Locally run Neural Networks
```ps
> netflow run my_network -f "image.png" --embedding rgbimage --decoder probability 
```

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
  

## Saved Data
All data is stored in the local application data directory `%LocalAppData%/DotML.NetFlow` on windows. You may specify a different directory to use for app storage using the `NETFLOW_HOME` environment variable.