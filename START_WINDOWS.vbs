Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "win.bat" & chr(34), 0
Set WshShell = Nothing