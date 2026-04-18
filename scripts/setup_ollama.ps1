# PowerShell script to set up Ollama models for Legal RAG

Write-Host "Legal RAG - Ollama Setup Script" -ForegroundColor Green
Write-Host "================================`n"

# Check if Ollama is installed
Write-Host "Checking if Ollama is installed..." -ForegroundColor Yellow
$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue

if (-not $ollamaPath) {
    Write-Host "❌ Ollama is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Ollama from: https://ollama.ai`n"
    exit 1
}

Write-Host "✓ Ollama found at: $($ollamaPath.Source)`n" -ForegroundColor Green

# Check if Ollama service is running
Write-Host "Checking if Ollama service is running..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -ErrorAction Stop
    Write-Host "✓ Ollama service is running`n" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama service is not running" -ForegroundColor Red
    Write-Host "Start Ollama with: ollama serve`n"
    exit 1
}

# Pull required models
$models = @("mistral", "nomic-embed-text")

foreach ($model in $models) {
    Write-Host "Pulling model: $model" -ForegroundColor Yellow
    & ollama pull $model
    Write-Host ""
}

# List available models
Write-Host "Available models:" -ForegroundColor Green
& ollama list

Write-Host "`n✓ Setup complete! Ready to use Legal RAG.`n" -ForegroundColor Green
