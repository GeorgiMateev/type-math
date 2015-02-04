#Runs the python script for spliting expressions into symbols for every .inkml file in the current directory. $st=[""/JUNK/BOTH]
param (
    [string] $st = ""
)
Get-ChildItem -Path $Pwd\* -include *.inkml -File | Foreach {$fileName="symbol-"+$_.name; py segGenerator.py $_.name $fileName $st}