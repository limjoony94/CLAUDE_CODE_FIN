' R26 Watchdog Silent Launcher
' -----------------------------
' Launches PowerShell watchdog with NO window visibility.
' Task Scheduler should run this .vbs via wscript.exe instead of powershell.exe directly.
'
' WScript.Shell.Run signature:
'   Run(command, windowStyle, waitForReturn)
'   windowStyle: 0 = hidden, 1 = normal, 2 = minimized, 7 = minimized no focus
'   waitForReturn: True = block until done, False = fire and forget

Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
psScript = scriptDir & "\r26_watchdog.ps1"

' Verify PS script exists
If Not fso.FileExists(psScript) Then
    WScript.Quit 1
End If

' Build command — quote path for spaces
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & psScript & """"

' Launch hidden (windowStyle=0), wait for completion (True)
Set shell = CreateObject("WScript.Shell")
shell.Run cmd, 0, True

Set shell = Nothing
Set fso = Nothing
