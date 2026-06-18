#define MyAppName "光环智能股票预测系统"
#define MyAppVersion "2.3.2"
#define MyAppPublisher "光环智能"
#define MyAppExeName "launch.bat"

[Setup]
AppId={{0E5BE9F8-5A52-4F0B-A0F7-1F50B92A2201}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=no
UsePreviousAppDir=no
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=.
OutputBaseFilename=光环智能股票预测系统_v{#MyAppVersion}_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\app_icon.ico
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "..\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc,*.bak,*.bak2,.cache\*,online_models\*,performance_logs\*,prediction_logs\*,history\*,output\*,scheduler\*,watchlist.json"
Source: "..\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "..\scripts\*"; DestDir: "{app}\scripts"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\launch.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\启动光环智能.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\启动桌面版.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\启动手机版.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\启动全部.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\启动回测.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\停止光环智能.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\查看日志.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\重启.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\使用说明.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\app_icon.ico"
Name: "{autoprograms}\停止{#MyAppName}"; Filename: "{app}\停止光环智能.bat"; WorkingDir: "{app}"; IconFilename: "{app}\app_icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\app_icon.ico"; Tasks: desktopicon
Name: "{autodesktop}\停止{#MyAppName}"; Filename: "{app}\停止光环智能.bat"; WorkingDir: "{app}"; IconFilename: "{app}\app_icon.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务："; Flags: checkedonce

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
