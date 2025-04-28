# PowerShell script to build and push Docker image for Home Assistant Anomaly Detection
# Usage: .\build-push.ps1 [version]

# Set default version if not specified
param(
    [string]$Version = "latest"
)

$ErrorActionPreference = "Stop"
$RepositoryName = "myevit/hass-anomaly-detection"
$FullImageName = "${RepositoryName}:${Version}"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Home Assistant Anomaly Detection Docker Build" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Building version: $Version" -ForegroundColor Green
Write-Host "Repository: $RepositoryName" -ForegroundColor Green
Write-Host ""

try {
    # Step 1: Build the Docker image
    Write-Host "Step 1: Building Docker image..." -ForegroundColor Yellow
    docker build -t $FullImageName .
    
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "Image built successfully!" -ForegroundColor Green
    Write-Host ""
    
    # Step 2: Log in to Docker Hub
    Write-Host "Step 2: Logging in to Docker Hub..." -ForegroundColor Yellow
    Write-Host "Please enter your Docker Hub credentials when prompted."
    
    docker login
    
    if ($LASTEXITCODE -ne 0) {
        throw "Docker login failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "Login successful!" -ForegroundColor Green
    Write-Host ""
    
    # Step 3: Push the image to Docker Hub
    Write-Host "Step 3: Pushing image to Docker Hub..." -ForegroundColor Yellow
    docker push $FullImageName
    
    if ($LASTEXITCODE -ne 0) {
        throw "Docker push failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "Image pushed successfully!" -ForegroundColor Green
    Write-Host ""
    
    # Tag with 'latest' if a specific version was given
    if ($Version -ne "latest") {
        Write-Host "Step 4: Tagging and pushing as 'latest' as well..." -ForegroundColor Yellow
        docker tag $FullImageName "${RepositoryName}:latest"
        docker push "${RepositoryName}:latest"
        
        if ($LASTEXITCODE -ne 0) {
            throw "Pushing latest tag failed with exit code $LASTEXITCODE"
        }
        
        Write-Host "Latest tag pushed successfully!" -ForegroundColor Green
        Write-Host ""
    }
    
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "         Docker Build/Push Complete!         " -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host "Image: $FullImageName" -ForegroundColor Green
    Write-Host ""
    Write-Host "To pull this image, run:" -ForegroundColor Yellow
    Write-Host "docker pull $FullImageName" -ForegroundColor White
    
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Build/push process failed!" -ForegroundColor Red
    exit 1
} 