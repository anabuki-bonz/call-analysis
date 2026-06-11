$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c cd /d "G:\マイドライブ\Claude Code\call-analysis" && git add . && git diff --cached --quiet || git commit -m "auto: daily backup" && git push'
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '18:00'
$settings = New-ScheduledTaskSettingsSet -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -RunLevel Highest -LogonType S4U
Register-ScheduledTask -TaskName 'GitAutoPush' -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force
