$ErrorActionPreference = 'Stop'
$backendPath = Join-Path $PSScriptRoot 'backend'
$frontendPath = Join-Path $PSScriptRoot 'frontend'
$runtimePath = Join-Path $PSScriptRoot 'data/runtime'
$modelPath = Join-Path $backendPath 'models/pose_landmarker_lite.task'
$vitePath = Join-Path $frontendPath 'node_modules/vite/bin/vite.js'
if (!(Test-Path -LiteralPath $modelPath) -or !(Test-Path -LiteralPath $vitePath)) {
    throw 'Faltan el modelo o las dependencias web. Sigue la preparación del README.'
}
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$savedPythonPath = $env:PYTHONPATH
try {
    $depsPath = Join-Path $backendPath '.deps'
    $venvPython = Join-Path $backendPath '.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $depsPath) {
        $env:PYTHONPATH = $depsPath
        $pythonCommand = (Get-Command python).Source
    } elseif (Test-Path -LiteralPath $venvPython) {
        $pythonCommand = $venvPython
    } else {
        throw 'Falta el entorno Python. Sigue la preparación del README.'
    }
    if (Get-NetTCPConnection -State Listen -LocalPort 8765 -ErrorAction SilentlyContinue) {
        $health = Invoke-RestMethod 'http://127.0.0.1:8765/api/health'
        if ($health.service -ne 'vision3d-ai') { throw 'El puerto 8765 pertenece a otro servicio.' }
    } else {
        Start-Process -FilePath $pythonCommand -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8765') -WorkingDirectory $backendPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'backend.log') -RedirectStandardError (Join-Path $runtimePath 'backend-error.log') | Out-Null
    }
    if (Get-NetTCPConnection -State Listen -LocalPort 5173 -ErrorAction SilentlyContinue) {
        $page = Invoke-WebRequest 'http://127.0.0.1:5173' -UseBasicParsing
        if ($page.Content -notmatch 'VISION3D') { throw 'El puerto 5173 pertenece a otra aplicación.' }
    } else {
        $nodeCommand = (Get-Command node).Source
        Start-Process -FilePath $nodeCommand -ArgumentList @(('"' + $vitePath + '"'), '--host', '127.0.0.1', '--port', '5173', '--strictPort') -WorkingDirectory $frontendPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimePath 'frontend.log') -RedirectStandardError (Join-Path $runtimePath 'frontend-error.log') | Out-Null
    }
} finally {
    $env:PYTHONPATH = $savedPythonPath
}
Write-Output 'Servicios iniciados en segundo plano. Abre http://127.0.0.1:5173'
Write-Output 'Registros locales: data/runtime. Si la página tarda, revisa backend-error.log.'
