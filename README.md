
brew install ffmpeg

https://github.com/oop7/ffmpeg-install-guide

powershell
# Откройте PowerShell от имени администратора
# Добавьте FFmpeg в системный PATH
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::Machine) + ";C:\ffmpeg\bin",
    [EnvironmentVariableTarget]::Machine
)


```
python main.py 'название.m4a' --model medium --device auto --compute-type auto

python main.py 'records\название.m4a' --model large-v3 --device auto --compute-type int8

```